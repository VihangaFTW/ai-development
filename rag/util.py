from dotenv import load_dotenv
import os
from enum import Enum


class Key(Enum):
    OPENAI  = "OPENAI_API_KEY"


def load_and_get_key() -> str | None:
    load_dotenv()
    return os.getenv(Key.OPENAI.value)
