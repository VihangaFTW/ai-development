from dotenv import load_dotenv
import os
import re
from enum import Enum


class Key(Enum):
    OPENAI  = "OPENAI_API_KEY"


def load_and_get_key() -> str | None:
    load_dotenv()
    return os.getenv(Key.OPENAI.value)


def word_wrap(text: str, line_width: int = 90) -> str:
    # normalize the text by removing existing line breaks and extra whitespace by
    # replacing multiple whitespace (including newlines) with single spaces
    normalized_text = re.sub(r'\s+', ' ', text.strip())
    
    # apply word wrapping to the normalized text
    return "\n".join([normalized_text[i:i+line_width] for i in range(0, len(normalized_text), line_width)])

