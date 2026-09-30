"""
사내 QC·재고 AI 어시스턴트 — Streamlit 채팅 UI.

실행: streamlit run app.py
"""

import os
import json
import hashlib

import streamlit as st

# set_page_config 는 반드시 첫 Streamlit 명령이어야 한다.
# st.secrets 접근도 Streamlit 명령으로 집계되므로 이 줄이 먼저 와야 한다.
st.set_page_config(page_title="QC·재고 AI 어시스턴트", page_icon="🔧", layout="centered")

# 사이드바의 '맨 위로' 링크가 가리킬 위치
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# 로컬은 .env, 배포 환경은 Secrets 를 사용한다.
# secrets.toml 이 없는 로컬에서는 st.secrets 접근 자체가 예외를 던지므로 감싼다.
from dotenv import load_dotenv
load_dotenv()

try:
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL", "EMBED_MODEL"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # secrets.toml 이 없으면 .env 값을 그대로 사용

import agent  # 키 설정 후에 import
import gemini_call

EXAMPLES_EXPERT = [
    "밴드형 기기 A 200대 생산하려는데 부속품 재고 충분해?",
    "안전재고 미달인 품목 알려줘",
    "밴드형 기기 A 출고 전 검사 항목이 뭐야?",
    "건식전극 재고 얼마 남았고, 부족하면 어디에 발주해야 해?",
]

EXAMPLES_ONBOARDING = [
    "BOM이 뭐예요?",
    "부속품이 입고되면 뭘 해야 하나요?",
    "불량이 나오면 어떤 순서로 처리하나요?",
    "안전재고 미달이면 바로 발주해야 하나요?",
]

# 도구 이름을 화면에 보여줄 문구로 옮긴다.
# 사용자는 함수명이 아니라 지금 무엇을 확인하는지를 알고 싶어한다.
STEP_LABELS = {
    "get_inventory": "재고를 확인하는 중",
    "get_bom": "BOM을 펼치는 중",
    "check_production_feasibility": "소요량과 재고를 대조하는 중",
    "get_supplier": "협력업체 정보를 찾는 중",
    "lookup_term": "용어를 찾아보는 중",
    "search_qc_docs": "QC 문서를 검색하는 중",
}


def step_text(name, args):
    label = STEP_LABELS.get(name, f"{name} 실행 중")
    detail = next(
        (str(v) for v in args.values() if isinstance(v, str) and v.strip()), ""
    )
    return f"{label} — {detail}" if detail else label


if "messages" not in st.session_state:
    st.session_state.messages = []
if "authed" not in st.session_state:
    st.session_state.authed = False


def check_password(raw):
    """설정된 해시와 비교한다. 평문 비밀번호는 어디에도 저장하지 않는다."""
    expected = os.environ.get("ADMIN_PASSWORD_HASH", "")
    if not expected:
        return None  # 미설정
    return hashlib.sha256(raw.encode()).hexdigest() == expected


def render_trace(trace):
    """어떤 도구가 어떤 인자로 호출됐고 무엇을 반환했는지 펼쳐볼 수 있게 표시한다."""
    if not trace:
        return
    with st.expander(f"도구 {len(trace)}회 호출됨", expanded=False):
        for t in trace:
            st.markdown(f"**{t['도구']}** `{json.dumps(t['인자'], ensure_ascii=False)}`")
            st.json(t["결과"], expanded=False)


# ---------------------------------------------------------------- 사이드바

with st.sidebar:
    if st.session_state.authed:
        st.success("로그인됨 — 품목명 전체 표시")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.authed = False
            st.rerun()
    else:
        with st.expander("🔒 관리자 로그인"):
            st.caption("로그인하면 가려진 품목명과 품번이 표시됩니다.")
            pw = st.text_input("비밀번호", type="password", key="pw")
            if st.button("로그인", use_container_width=True):
                result = check_password(pw)
                if result is None:
                    st.error("서버에 비밀번호가 설정되어 있지 않습니다.")
                elif result:
                    st.session_state.authed = True
                    st.rerun()
                else:
                    st.error("비밀번호가 맞지 않습니다.")

    st.divider()
    onboarding = st.toggle(
        "신입 모드",
        value=False,
        help="용어 뜻과 판단 근거를 함께 설명합니다. 처음 업무를 배울 때 사용하세요.",
    )

    st.divider()
    st.subheader("예시 질문")
    examples = EXAMPLES_ONBOARDING if onboarding else EXAMPLES_EXPERT
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

    st.divider()
    st.subheader("연결된 도구")
    st.markdown(
        "- `get_inventory` 재고 조회\n"
        "- `get_bom` BOM 전개·소요량\n"
        "- `check_production_feasibility` 생산 가능성 판정\n"
        "- `get_supplier` 협력업체 조회\n"
        "- `lookup_term` 현장 용어 조회\n"
        "- `search_qc_docs` QC 문서 검색 (RAG)"
    )

    st.divider()
    st.markdown(
        '<a href="#top" style="text-decoration:none;">⬆ 맨 위로</a>',
        unsafe_allow_html=True,
    )
    st.caption(
        "데이터는 포트폴리오용 가상 값입니다. "
        "실제 사내 데이터와 문서는 포함되어 있지 않습니다."
    )


# ---------------------------------------------------------------- 본문

st.title("QC·재고 AI 어시스턴트")
if onboarding:
    st.caption(
        "신입 모드입니다. 용어의 뜻과 그렇게 판단하는 이유를 함께 설명합니다."
    )
else:
    st.caption(
        "사내 재고 대시보드 데이터와 QC 문서를 함께 조회하는 챗봇입니다. "
        "질문에 따라 필요한 도구를 스스로 선택합니다."
    )

if not st.session_state.authed:
    st.warning(
        "로그인 전에는 품목명과 품번이 가려집니다 (예: FP-1**, 배터리 셀(4*****)). "
        "나머지 수량·업체·검사 기준은 그대로 확인할 수 있습니다."
    )

if not st.session_state.messages:
    st.info("왼쪽 사이드바의 예시 질문을 누르거나, 아래에 직접 물어보세요.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_trace(msg.get("trace"))
        st.markdown(msg["content"])
        if msg.get("from_docs"):
            st.caption("검사 기준은 시연용으로 작성한 가상 문서 기준입니다.")

# 마지막 답변의 후속 질문만 버튼으로 노출한다
last = st.session_state.messages[-1] if st.session_state.messages else None
if last and last["role"] == "assistant" and last.get("followups"):
    st.markdown("**이어서 물어보기**")
    cols = st.columns(len(last["followups"]))
    for i, q in enumerate(last["followups"]):
        if cols[i].button(q, key=f"fu{len(st.session_state.messages)}_{i}",
                          use_container_width=True):
            st.session_state.pending = q
            st.rerun()

# 대화가 있을 때만 초기화 버튼을 입력창 바로 위에 둔다
if st.session_state.messages:
    _, right = st.columns([5, 1])
    if right.button("🗑 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

prompt = st.chat_input("재고, BOM, 검사 기준에 대해 물어보세요")
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 도구 호출 기록도 함께 넘긴다. 어디까지 문맥에 실을지는 agent 가 정한다.
    history = []
    for m in st.session_state.messages[:-1]:
        entry = {
            "role": "model" if m["role"] == "assistant" else "user",
            "text": m["content"],
        }
        if m["role"] == "assistant" and m.get("trace"):
            entry["trace"] = m["trace"]
        history.append(entry)

    with st.chat_message("assistant"):
        with st.status("질문을 살펴보는 중", expanded=True) as status:

            def on_step(name, args):
                text = step_text(name, args)
                status.update(label=text)
                st.write(text)

            try:
                answer, trace, followups = agent.run(
                    prompt,
                    history,
                    onboarding,
                    masked=not st.session_state.authed,
                    on_step=on_step,
                )
                status.update(label="확인 완료", state="complete", expanded=False)
            except gemini_call.GeminiUserError as e:
                # 서버 붐빔·사용 한도·설정 오류: 한국어 안내만 (원문은 서버 로그에 이미 남김)
                answer = e.user_message
                trace, followups = [], []
                status.update(label="처리 중단됨", state="error", expanded=False)
            except Exception as e:
                # 예상하지 못한 오류도 원문은 화면에 내보내지 않는다
                gemini_call.log_error("질문 처리", e)
                answer = gemini_call.MSG_OTHER
                trace, followups = [], []
                status.update(label="처리 중단됨", state="error", expanded=False)

        from_docs = any(t["도구"] == "search_qc_docs" for t in trace)

        render_trace(trace)
        st.markdown(answer)
        if from_docs:
            st.caption("검사 기준은 시연용으로 작성한 가상 문서 기준입니다.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
        "from_docs": from_docs,
        "followups": followups,
    })
    st.rerun()  # 후속 질문 버튼을 즉시 표시하기 위해 다시 그린다
