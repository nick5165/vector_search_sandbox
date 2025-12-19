import re
from typing import List

def tokenize_text(text: str) -> List[str]:
    """
    Базовая токенизация для BM25
    """
    
    return re.findall(r"(?u)\b\w+\b", text.lower())

def split_into_sentences(text: str) -> List[str]:
    """
    Разбиение на предложения
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]