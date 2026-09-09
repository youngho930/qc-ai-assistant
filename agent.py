"""
Gemini function calling 기반 에이전트.

모델에게 도구 5개를 쥐여주고, 질문에 따라 알아서 고르게 합니다.
SDK의 자동 호출 기능을 끄고 루프를 직접 돌리는 이유는
어떤 도구가 어떤 인자로 불렸는지 UI에 그대로 보여주기 위해서입니다.
"""

import os
import json

from google import genai
from google.genai import types

import tools
import rag

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_STEPS = 6

SYSTEM_PROMPT = """\
너는 의료기기 제조사의 생산·품질 담당자를 돕는 사내 어시스턴트다.

원칙:
- 재고, BOM, 협력업체, 검사 기준에 대한 질문은 반드시 도구를 호출해 확인한 뒤 답한다.
  기억이나 추측으로 수치를 말하지 않는다.
- 한 질문에 여러 정보가 필요하면 도구를 여러 번 호출한다.
  예: "200대 생산 가능한지"는 생산 가능성 판정을, "검사 기준"은 문서 검색을 쓴다.
- 문서 내용을 인용할 때는 어느 문서의 어느 항목인지 밝힌다.
- 도구가 결과를 찾지 못했으면 모른다고 답한다. 없는 기준을 지어내지 않는다.
- 수치는 표로 정리하면 읽기 쉽다. 답변은 간결하게, 실무자가 바로 판단할 수 있게 쓴다.
"""

ALL_SCHEMAS = tools.TOOL_SCHEMAS + [rag.SEARCH_SCHEMA]
ALL_FUNCS = {**tools.TOOL_FUNCS, "search_qc_docs": rag.search_qc_docs}


def _client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY 가 설정되지 않았습니다.")
    return genai.Client(api_key=key)


def _config():
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=ALL_SCHEMAS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )


def run(user_message, history=None):
    """
    질문 하나를 처리한다.

    history: [{"role": "user"/"model", "text": ...}] 형태의 이전 대화
    반환: (답변 문자열, 호출된 도구 기록 리스트)
    """
    client = _client()

    contents = []
    for turn in history or []:
        contents.append(types.Content(
            role=turn["role"],
            parts=[types.Part.from_text(text=turn["text"])],
        ))
    contents.append(types.Content(
        role="user", parts=[types.Part.from_text(text=user_message)]
    ))

    trace = []

    for _ in range(MAX_STEPS):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=_config()
        )

        candidate = response.candidates[0]
        calls = [p.function_call for p in (candidate.content.parts or [])
                 if getattr(p, "function_call", None)]

        if not calls:
            return (response.text or "").strip(), trace

        contents.append(candidate.content)

        # 모델이 한 턴에 여러 도구를 부를 수 있으므로 전부 처리한다
        result_parts = []
        for call in calls:
            args = dict(call.args or {})
            try:
                output = ALL_FUNCS[call.name](**args)
            except Exception as e:
                output = {"오류": f"{type(e).__name__}: {e}"}

            trace.append({"도구": call.name, "인자": args, "결과": output})
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": output}
                )
            )

        contents.append(types.Content(role="user", parts=result_parts))

    return "도구 호출이 반복돼 응답을 마치지 못했습니다. 질문을 나눠서 물어봐 주세요.", trace
