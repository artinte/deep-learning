import tiktoken
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Tokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens"""

    encoding: tiktoken.Encoding
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)
