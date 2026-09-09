"""
QC 문서 임베딩 색인 생성 스크립트.

data/docs/*.md 를 수정한 뒤 한 번 실행하면 index/docs_index.npz 가 갱신됩니다.
생성된 색인은 저장소에 커밋해두면 배포 시 재계산이 필요 없습니다.

    python build_index.py
"""

from dotenv import load_dotenv
load_dotenv()

import rag

if __name__ == "__main__":
    rag.build_index()
