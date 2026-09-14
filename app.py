"""
사내 QC·재고 AI 어시스턴트 — Streamlit 채팅 UI.

실행: streamlit run app.py
"""

import os
import json

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
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "EMBED_MODEL"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # secrets.toml 이 없으면 .env 값을 그대로 사용

import agent  # 키 설정 후에 import

EXAMPLES = [
    "S26 밴드 200대 생산하려는데 부속품 재고 충분해?",
    "안전재고 미달인 품목 알려줘",
    "S26 밴드 출고 전 검사 항목이 뭐야?",
    "건식전극 재고 얼마 남았고, 부족하면 어디에 발주해야 해?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []


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
    st.subheader("연결된 도구")
    st.markdown(
        "- `get_inventory` 재고 조회\n"
        "- `get_bom` BOM 전개·소요량\n"
        "- `check_production_feasibility` 생산 가능성 판정\n"
        "- `get_supplier` 협력업체 조회\n"
        "- `search_qc_docs` QC 문서 검색 (RAG)"
    )

    st.divider()
    st.subheader("예시 질문")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

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
st.caption(
    "사내 재고 대시보드 데이터와 QC 문서를 함께 조회하는 챗봇입니다. "
    "질문에 따라 필요한 도구를 스스로 선택합니다."
)

if not st.session_state.messages:
    st.info("왼쪽 사이드바의 예시 질문을 누르거나, 아래에 직접 물어보세요.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_trace(msg.get("trace"))
        st.markdown(msg["content"])
        if msg.get("from_docs"):
            st.caption("검사 기준은 시연용으로 작성한 가상 문서 기준입니다.")

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

    history = [
        {"role": "model" if m["role"] == "assistant" else "user", "text": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("확인하는 중..."):
            try:
                answer, trace = agent.run(prompt, history)
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    answer = (
                        "무료 API 사용 한도에 도달했습니다. "
                        "잠시 후 다시 시도해 주세요."
                    )
                else:
                    answer = f"오류가 발생했습니다: {e}"
                trace = []

        from_docs = any(t["도구"] == "search_qc_docs" for t in trace)

        render_trace(trace)
        st.markdown(answer)
        if from_docs:
            st.caption("검사 기준은 시연용으로 작성한 가상 문서 기준입니다.")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "trace": trace, "from_docs": from_docs}
    )