"""
Gemini 호출 공통 처리: 일시 오류 재시도, 예비 모델, 사용자용 한국어 안내.

- 일시 오류(429, 500, 502, 503, 504)만 간격을 늘려가며 최대 3번 다시 시도한다.
- 인증 오류(401, 403)와 모델 이름 오류(404)는 다시 시도해도 소용없으므로 바로 멈춘다.
- 화면에는 한국어 안내만 보여주고, 에러 원문은 서버 로그에만 남긴다 (API 키는 지운다).
"""

import logging
import os
import re
import sys
import time

from google.genai import errors

TRANSIENT_CODES = {429, 500, 502, 503, 504}
CONFIG_CODES = {401, 403, 404}
RETRY_DELAYS = (1, 2, 4)  # 초. 다시 시도할 때마다 간격을 늘린다 (최대 3번)

MSG_BUSY = "AI 서버가 잠시 붐빕니다. 잠시 후 다시 시도해 주세요."
MSG_QUOTA = "무료 API 사용 한도에 도달했습니다. 잠시 후 다시 시도해 주세요."
MSG_CONFIG = "AI 설정 오류로 답변할 수 없습니다. 관리자에게 문의해 주세요."
MSG_OTHER = "답변을 만드는 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요."

log = logging.getLogger("qc_ai")
if not log.handlers:  # Streamlit 로그(표준 에러)로 나가게 한다
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")

# 테스트에서 기다리지 않도록 바꿀 수 있게 둔다
sleep = time.sleep


class GeminiUserError(Exception):
    """화면에 그대로 보여줘도 되는 한국어 안내를 담은 오류."""

    def __init__(self, user_message):
        super().__init__(user_message)
        self.user_message = user_message


class GeminiUnavailable(GeminiUserError):
    """다시 시도와 예비 모델까지 실패한 일시 오류."""


class GeminiConfigError(GeminiUserError):
    """키·권한·모델 이름 같은 설정 문제. 다시 시도하지 않는다."""

    def __init__(self):
        super().__init__(MSG_CONFIG)


def safe_text(text):
    """로그에 남기기 전에 API 키로 보이는 값을 지운다."""
    text = str(text)
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        text = text.replace(key, "[API 키 가림]")
    return _KEY_PATTERN.sub("[API 키 가림]", text)


def log_error(label, error):
    """에러 원문은 서버 로그에만 (키는 지운 뒤)."""
    code = getattr(error, "code", None)
    log.warning("%s 실패 (%s %s): %s", label, code or "-", type(error).__name__, safe_text(error)[:500])


def with_retry(label, call):
    """
    call() 을 실행한다. 일시 오류면 RETRY_DELAYS 간격으로 최대 3번 다시 시도한다.
    - 설정 오류(401/403/404): 바로 GeminiConfigError
    - 끝내 일시 오류: 마지막 APIError 를 그대로 올려, 호출한 쪽이 예비 모델을 쓸지 정한다
    - 그 밖의 오류: 그대로 올린다
    """
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return call()
        except errors.APIError as e:
            log_error(f"{label} (시도 {attempt + 1})", e)
            if e.code in CONFIG_CODES:
                raise GeminiConfigError() from None
            if e.code not in TRANSIENT_CODES or attempt == len(RETRY_DELAYS):
                raise
            sleep(RETRY_DELAYS[attempt])


def unavailable_from(error):
    """끝내 실패한 일시 오류를 사용자 안내로 바꾼다 (429 는 사용 한도 안내)."""
    return GeminiUnavailable(MSG_QUOTA if getattr(error, "code", None) == 429 else MSG_BUSY)
