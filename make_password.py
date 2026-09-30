"""
관리자 비밀번호 해시 생성 스크립트.

    python make_password.py

출력된 해시를 .env 의 ADMIN_PASSWORD_HASH 에 넣으세요.
평문 비밀번호는 저장하지 않습니다.
"""

import getpass
import hashlib

if __name__ == "__main__":
    pw = getpass.getpass("설정할 비밀번호: ")
    if not pw:
        raise SystemExit("비밀번호가 비어 있습니다.")
    if pw != getpass.getpass("한 번 더 입력: "):
        raise SystemExit("두 입력이 일치하지 않습니다.")

    print()
    print("아래 줄을 .env 에 추가하세요.")
    print(f"ADMIN_PASSWORD_HASH={hashlib.sha256(pw.encode()).hexdigest()}")
