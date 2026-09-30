"""
Gemini function calling 기반 에이전트.

모델에게 도구 6개를 쥐여주고, 질문에 따라 알아서 고르게 합니다.
SDK의 자동 호출 기능을 끄고 루프를 직접 돌리는 이유는
어떤 도구가 어떤 인자로 불렸는지 UI에 그대로 보여주기 위해서입니다.
"""

import os
import re
import json

from google import genai
from google.genai import types

import tools
import rag
import mask
import gemini_call

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_STEPS = 6

# 도구 결과를 문맥에 남길 직전 턴 수.
# 전부 남기면 "그럼 200대는?" 같은 후속 질문이 정확해지지만,
# BOM 전개처럼 긴 JSON이 쌓여 무료 티어 한도를 몇 턴 만에 소진한다.
# 실제로 이어받아야 하는 것은 대부분 바로 앞 턴이라 1로 두었다.
CONTEXT_TURNS = 1
TRACE_BUDGET = 1200  # 한 턴의 도구 결과를 문맥에 넣을 때의 글자 상한

# 같은 데이터라도 묻는 사람에 따라 필요한 설명이 다르다.
# 실무자에게는 결론이, 처음 배우는 사람에게는 맥락이 먼저 필요하다.

COMMON_RULES = """\
원칙:
- 재고, BOM, 협력업체, 검사 기준, 용어에 대한 질문은 반드시 도구를 호출해 확인한 뒤 답한다.
  기억이나 추측으로 수치와 정의를 말하지 않는다.
- 한 질문에 여러 정보가 필요하면 도구를 여러 번 호출한다.
- 문서 내용을 인용할 때는 어느 문서의 어느 항목인지 밝힌다.
- 도구가 결과를 찾지 못했으면 모른다고 답한다. 없는 기준을 지어내지 않는다.
- 재고를 수정하거나 발주를 실행하는 기능은 없다. 요청받으면 조회만 가능하다고 답한다.

답변 맨 마지막 줄에는 이어서 물어볼 만한 질문 2개를 아래 형식으로 덧붙인다.
이 줄은 사용자에게 버튼으로 표시되므로 형식을 정확히 지킨다.

[다음질문] 첫 번째 질문 | 두 번째 질문

질문은 실제로 이 도구가 답할 수 있는 범위여야 하며,
방금 답한 내용에서 자연스럽게 이어지는 것이어야 한다.
"""

PROMPT_EXPERT = """\
너는 의료기기 제조사의 생산·품질 담당자를 돕는 사내 어시스턴트다.
상대는 현장 실무자이므로 용어를 따로 풀어주지 않아도 된다.
수치는 표로 정리하고, 실무자가 바로 판단할 수 있게 간결히 쓴다.

""" + COMMON_RULES

PROMPT_ONBOARDING = """\
너는 의료기기 제조사 생산품질팀에 막 배치된 신입 담당자를 돕는 사내 어시스턴트다.
상대는 현장 용어와 업무 순서를 아직 모른다.

신입을 대할 때 지켜야 할 것:
- 답변에 현장 용어가 나오면 짧게 뜻을 덧붙인다. 확실하지 않으면 lookup_term 으로 확인한다.
- 수치만 주지 말고 그 수치가 왜 그런지, 어기면 무슨 일이 생기는지 함께 설명한다.
- 판단이 필요한 상황이면 혼자 결정하지 말고 확인받아야 할 지점을 알려준다.
- 문장은 쉽게 쓰되 기준 자체는 문서 그대로 정확히 전달한다. 쉽게 쓰려고 기준을 바꾸지 않는다.

""" + COMMON_RULES

ALL_SCHEMAS = tools.TOOL_SCHEMAS + [rag.SEARCH_SCHEMA]
ALL_FUNCS = {**tools.TOOL_FUNCS, "search_qc_docs": rag.search_qc_docs}

FOLLOWUP_RE = re.compile(r"\[다음질문\]\s*(.+)\s*$")


def summarize_trace(trace, budget=TRACE_BUDGET):
    """직전 턴의 도구 결과를 문맥에 넣을 수 있게 압축한다.

    전체를 그대로 넣으면 토큰이 빠르게 불어나므로, 도구별로 글자 수를 나눠
    상한까지만 싣는다. 잘린 부분은 필요하면 모델이 도구를 다시 호출한다.
    """
    if not trace:
        return ""

    share = max(200, budget // len(trace))
    lines = []
    for t in trace:
        args = json.dumps(t["인자"], ensure_ascii=False)
        body = json.dumps(t["결과"], ensure_ascii=False)
        if len(body) > share:
            body = body[:share] + " …(생략)"
        lines.append(f"- {t['도구']}{args} → {body}")

    return (
        "(직전 질문에서 도구로 확인한 값이다. "
        "이어지는 질문의 맥락으로만 참고하고, 최신 값이 필요하면 다시 호출한다.)\n"
        + "\n".join(lines)
    )


def _client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        gemini_call.log.warning("GEMINI_API_KEY 가 설정되지 않았습니다.")
        raise gemini_call.GeminiConfigError()
    return genai.Client(api_key=key)


def _generate(client, contents, config):
    """
    답변 생성. 일시 오류는 기본 모델로 최대 3번 다시 시도하고,
    그래도 안 되면 예비 모델(GEMINI_FALLBACK_MODEL)이 설정돼 있을 때만 한 번 더 시도한다.
    """
    call = lambda model: client.models.generate_content(model=model, contents=contents, config=config)
    try:
        return gemini_call.with_retry(f"generate_content[{MODEL}]", lambda: call(MODEL))
    except gemini_call.errors.APIError as e:
        if e.code not in gemini_call.TRANSIENT_CODES:
            gemini_call.log_error("generate_content (재시도 안 하는 오류)", e)
            raise gemini_call.GeminiUserError(gemini_call.MSG_OTHER) from None
        last = e

    fallback = os.environ.get("GEMINI_FALLBACK_MODEL", "").strip()
    if fallback and fallback != MODEL:
        gemini_call.log.warning("기본 모델이 계속 실패해 예비 모델로 한 번 더 시도합니다.")
        try:
            return call(fallback)
        except gemini_call.errors.APIError as e:
            gemini_call.log_error(f"generate_content[예비 {fallback}]", e)
            if e.code in gemini_call.CONFIG_CODES:
                raise gemini_call.GeminiConfigError() from None
            last = e
    raise gemini_call.unavailable_from(last) from None


def _config(onboarding, masked):
    prompt = PROMPT_ONBOARDING if onboarding else PROMPT_EXPERT
    if masked:
        prompt += "\n" + mask.MASK_NOTICE
    return types.GenerateContentConfig(
        system_instruction=prompt,
        tools=[types.Tool(function_declarations=ALL_SCHEMAS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )


def _split_followups(text):
    """답변 끝에 붙은 후속 질문 줄을 분리한다. 없으면 빈 목록을 돌려준다."""
    lines = text.rstrip().split("\n")
    match = FOLLOWUP_RE.match(lines[-1].strip()) if lines else None
    if not match:
        return text.strip(), []

    items = [q.strip() for q in match.group(1).split("|")]
    items = [q for q in items if 3 < len(q) < 60][:2]
    return "\n".join(lines[:-1]).strip(), items


def run(user_message, history=None, onboarding=False, masked=False, on_step=None):
    """
    질문 하나를 처리한다.

    history: [{"role": "user"/"model", "text": ...}] 형태의 이전 대화
    onboarding: True 면 신입용 설명 방식으로 답한다
    masked: True 면 품목명·품번을 가린 값만 모델에 전달한다
    on_step: 도구를 호출하기 직전에 (도구명, 인자) 로 불리는 콜백.
             에이전트가 UI를 직접 건드리지 않도록 표시는 호출한 쪽에 맡긴다.
    반환: (답변, 호출된 도구 기록, 후속 질문 목록)
    """
    client = _client()

    turns = history or []

    # 도구 결과를 붙일 대상: 가장 최근 model 턴 CONTEXT_TURNS 개
    model_idx = [i for i, t in enumerate(turns) if t["role"] == "model"]
    with_trace = set(model_idx[-CONTEXT_TURNS:]) if CONTEXT_TURNS else set()

    contents = []
    for i, turn in enumerate(turns):
        text = turn["text"]
        if i in with_trace and turn.get("trace"):
            summary = summarize_trace(turn["trace"])
            if summary:
                text = f"{text}\n\n{summary}"
        contents.append(types.Content(
            role=turn["role"],
            parts=[types.Part.from_text(text=text)],
        ))
    contents.append(types.Content(
        role="user", parts=[types.Part.from_text(text=user_message)]
    ))

    trace = []

    for _ in range(MAX_STEPS):
        response = _generate(client, contents, _config(onboarding, masked))

        candidate = response.candidates[0]
        calls = [p.function_call for p in (candidate.content.parts or [])
                 if getattr(p, "function_call", None)]

        if not calls:
            answer, followups = _split_followups(response.text or "")
            return answer, trace, followups

        contents.append(candidate.content)

        # 모델이 한 턴에 여러 도구를 부를 수 있으므로 전부 처리한다
        result_parts = []
        for call in calls:
            args = dict(call.args or {})

            if on_step:
                try:
                    on_step(call.name, args)
                except Exception:
                    pass  # 표시 실패가 본 로직을 막지 않게 한다

            try:
                output = ALL_FUNCS[call.name](**args)
            except Exception as e:
                output = {"오류": f"{type(e).__name__}: {e}"}

            # 화면 표시 전이 아니라 모델에 넘기기 전에 가린다.
            # 그래야 모델이 원본을 보고 설명 문장에 흘리는 일이 없다.
            if masked:
                output = mask.apply(output)

            trace.append({"도구": call.name, "인자": args, "결과": output})
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": output}
                )
            )

        contents.append(types.Content(role="user", parts=result_parts))

    return "도구 호출이 반복돼 응답을 마치지 못했습니다. 질문을 나눠서 물어봐 주세요.", trace, []
