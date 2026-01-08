import logging
import os
from typing import Optional

try:
    import tiktoken
except Exception:
    tiktoken = None

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENCODING_NAME = os.getenv("TOKEN_ENCODING", "cl100k_base")
TOKEN_LIMIT = int(os.getenv("LLM_TOKEN_LIMIT", "6000"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("youtube_summarizer")


def count_tokens(text: str) -> Optional[int]:
    """Return an approximate token count for the given text.

    Returns None if `tiktoken` is not available.
    """
    if tiktoken is None:
        return None
    try:
        enc = tiktoken.get_encoding(ENCODING_NAME)
    except Exception:
        try:
            enc = tiktoken.encoding_for_model(ENCODING_NAME)
        except Exception:
            # fallback to a common encoding
            enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
