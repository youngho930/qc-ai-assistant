"""
품목 식별자 마스킹.

로그인하지 않은 상태에서는 도구가 돌려준 데이터의 품목명·품번을 가립니다.

가리는 위치를 UI가 아니라 도구 반환값으로 잡은 이유:
화면에서만 가리면 모델은 원본을 그대로 보게 되어, 표에는 마스킹된 값이 나와도
설명 문장에 원래 이름을 그대로 쓸 수 있습니다. 데이터 단계에서 가리면
모델도 가려진 값만 보므로 새어나갈 경로가 없습니다.

주의: 이것은 접근 제어가 아니라 표시 제어입니다.
실제로 보호가 필요한 데이터라면 서버에서 권한을 확인해 걸러야 하며,
공개 저장소에 원본을 두어서는 안 됩니다.
"""

import re

# 숫자를 포함한 영숫자 덩어리를 모델 식별자로 본다.
# S26 -> S**,  105 -> ***,  DTP503030 -> D********
# 한글 일반 명사(밴드, 건식전극)는 남긴다. 제품을 특정하는 것은 모델 번호 쪽이고,
# 범주명까지 지우면 답변을 읽을 수 없게 되기 때문이다.
TOKEN = re.compile(r"(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{2,}")

# 품목과 무관한 값까지 가리지 않도록, 마스킹할 필드를 지정한다.
TARGET_KEYS = {"품번", "품목명", "완제품", "등록된_용어", "내용", "항목", "출처"}


def _mask_token(m):
    t = m.group(0)
    return t[0] + "*" * (len(t) - 1)


def mask_text(text):
    return TOKEN.sub(_mask_token, text)


def apply(obj, inside_target=False):
    """도구 반환값을 재귀적으로 훑어 품목 식별자를 가린다."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = apply(v, inside_target or k in TARGET_KEYS)
        return out

    if isinstance(obj, list):
        return [apply(v, inside_target) for v in obj]

    if isinstance(obj, str) and inside_target:
        return mask_text(obj)

    return obj


MASK_NOTICE = (
    "품목명과 품번은 로그인 전이라 일부 가려져 있다. "
    "가려진 부분을 추측해 복원하려 하지 말고, 받은 값 그대로 사용한다. "
    "사용자가 질문에서 가려지지 않은 이름을 썼더라도, 답변에는 반드시 "
    "도구가 돌려준 가려진 표기를 사용한다. "
    "사용자가 원래 이름을 물으면 로그인이 필요하다고 안내한다."
)