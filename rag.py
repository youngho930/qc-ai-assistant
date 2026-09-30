"""
QC 문서 검색 (RAG).

data/docs/*.md 를 문단 단위로 쪼개 Gemini 임베딩으로 색인하고,
질문과 코사인 유사도가 높은 문단을 반환합니다.

벡터 DB를 쓰지 않는 이유:
문서가 수백 문단 규모라 numpy 내적 한 번이면 충분하고,
Chroma/FAISS 의존성을 빼면 Streamlit Cloud 배포가 훨씬 가볍고 안정적입니다.
"""

import os
import json
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

import gemini_call

BASE = Path(__file__).parent
DOCS_DIR = BASE / "data" / "docs"
INDEX_PATH = BASE / "index" / "docs_index.npz"

EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")


def _client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        gemini_call.log.warning("GEMINI_API_KEY 가 설정되지 않았습니다.")
        raise gemini_call.GeminiConfigError()
    return genai.Client(api_key=key)


# ---------------------------------------------------------------- 문서 로드

def load_chunks():
    """문서를 문단 단위로 쪼갠다. 헤딩을 만나면 이후 문단에 제목을 붙여둔다."""
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        heading = ""
        buf = []

        def flush():
            if buf:
                body = "\n".join(buf).strip()
                if len(body) > 20:  # 너무 짧은 조각은 검색 노이즈만 된다
                    chunks.append({
                        "source": path.name,
                        "heading": heading,
                        "text": f"[{path.stem} / {heading}]\n{body}" if heading else body,
                    })
            buf.clear()

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                flush()
                heading = line.lstrip("#").strip()
            elif not line.strip():
                flush()
            else:
                buf.append(line)
        flush()
    return chunks


# ---------------------------------------------------------------- 색인 생성

def _embed(client, texts, task_type):
    """임베딩 호출. 무료 티어 분당 요청 제한을 고려해 배치로 보낸다."""
    vectors = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i + 20]
        # 일시 오류는 다시 시도. 임베딩은 모델을 바꾸면 벡터가 달라지므로 예비 모델은 쓰지 않는다.
        # 끝내 실패하면 원문 대신 한국어 안내 오류를 올린다 (도구 결과로 모델에 전달되므로).
        try:
            res = gemini_call.with_retry(
                f"embed_content[{EMBED_MODEL}]",
                lambda: client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                ),
            )
        except gemini_call.errors.APIError as e:
            if e.code in gemini_call.TRANSIENT_CODES:
                raise gemini_call.unavailable_from(e) from None
            gemini_call.log_error("embed_content (재시도 안 하는 오류)", e)
            raise gemini_call.GeminiUserError(gemini_call.MSG_OTHER) from None
        vectors.extend([e.values for e in res.embeddings])
    arr = np.array(vectors, dtype=np.float32)
    # 정규화해두면 검색 시 내적만으로 코사인 유사도가 나온다
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def build_index():
    """문서를 임베딩해 index/docs_index.npz 로 저장한다. 문서 수정 시 다시 실행."""
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"{DOCS_DIR} 에 .md 문서가 없습니다.")

    client = _client()
    vecs = _embed(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")

    INDEX_PATH.parent.mkdir(exist_ok=True)
    np.savez(
        INDEX_PATH,
        vectors=vecs,
        meta=np.array(json.dumps(chunks, ensure_ascii=False)),
    )
    print(f"색인 완료: 문단 {len(chunks)}개 -> {INDEX_PATH}")
    return len(chunks)


# ---------------------------------------------------------------- 검색

_cache = None


def _load_index():
    global _cache
    if _cache is None:
        if not INDEX_PATH.exists():
            raise RuntimeError(
                "색인 파일이 없습니다. 먼저 `python build_index.py` 를 실행하세요."
            )
        data = np.load(INDEX_PATH, allow_pickle=False)
        _cache = (data["vectors"], json.loads(str(data["meta"])))
    return _cache


def search_qc_docs(query: str, top_k: int = 3) -> dict:
    """QC 문서에서 질문과 관련된 문단을 찾아 근거와 함께 반환한다."""
    vectors, chunks = _load_index()
    qv = _embed(_client(), [query], "RETRIEVAL_QUERY")[0]

    scores = vectors @ qv
    idx = np.argsort(-scores)[:top_k]

    hits = []
    for i in idx:
        if scores[i] < 0.35:  # 관련 없는 문단을 억지로 붙이지 않는다
            continue
        hits.append({
            "출처": chunks[i]["source"],
            "항목": chunks[i]["heading"],
            "내용": chunks[i]["text"],
            "유사도": round(float(scores[i]), 3),
        })

    if not hits:
        return {
            "결과": [],
            "안내": "관련 문서를 찾지 못했습니다. 문서에 없는 내용은 추측하지 말고 "
                    "'해당 내용은 등록된 문서에 없습니다' 라고 답하세요.",
        }
    return {"결과": hits}


SEARCH_SCHEMA = {
    "name": "search_qc_docs",
    "description": (
        "사내 QC 문서(검사기준서, 작업지침, 불량 판정 기준 등)에서 "
        "질문과 관련된 내용을 찾는다. 검사 방법, 판정 기준, 절차, "
        "규격에 대한 질문에 사용한다."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "문서에서 찾을 내용. 사용자 질문을 그대로 넣어도 된다.",
            }
        },
        "required": ["query"],
    },
}
