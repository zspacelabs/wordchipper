from dataclasses import dataclass
from typing import Any

HF_TO_WORDCHIPPER: dict[str, str]

@dataclass
class Encoding:
    ids: list[int]
    tokens: list[str]

class Tokenizer:
    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs: Any) -> Tokenizer: ...
    def encode(
        self,
        sequence: str,
        pair: str | None = ...,
        is_pretokenized: bool = ...,
        add_special_tokens: bool = ...,
    ) -> Encoding: ...
    def encode_batch(
        self,
        input: list[str],
        is_pretokenized: bool = ...,
        add_special_tokens: bool = ...,
    ) -> list[Encoding]: ...
    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = ...,
    ) -> str: ...
    def decode_batch(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = ...,
    ) -> list[str]: ...
    def get_vocab_size(self, with_added_tokens: bool = ...) -> int: ...
    def token_to_id(self, token: str) -> int | None: ...
    def id_to_token(self, id: int) -> str | None: ...
