"""Drop-in replacement for the HuggingFace ``tokenizers`` library.

Typical migration::

    # Before
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained("Xenova/gpt-4o")

    # After
    from wordchipper.compat.tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained("Xenova/gpt-4o")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wordchipper import Tokenizer as _WCTokenizer

# ---------------------------------------------------------------------------
# HuggingFace identifier -> wordchipper encoding name
# ---------------------------------------------------------------------------

HF_TO_WORDCHIPPER: dict[str, str] = {
    "Xenova/gpt-4o": "o200k_base",
    "Xenova/gpt-4": "cl100k_base",
    "Xenova/cl100k_base": "cl100k_base",
    "Xenova/o200k_base": "o200k_base",
    "Xenova/text-davinci-003": "p50k_base",
    "Xenova/text-embedding-ada-002": "cl100k_base",
}


@dataclass
class Encoding:
    """Result of a single encode call (mirrors ``tokenizers.Encoding``)."""

    ids: list[int]
    tokens: list[str]


class Tokenizer:
    """Wrapper around :class:`wordchipper.Tokenizer` with HuggingFace's API."""

    def __init__(self, inner: _WCTokenizer) -> None:
        self._tok = inner

    @classmethod
    def from_pretrained(cls, identifier: str, **_kwargs: Any) -> Tokenizer:
        """Load a tokenizer by HuggingFace identifier or bare encoding name."""
        name = HF_TO_WORDCHIPPER.get(identifier, identifier)
        return cls(_WCTokenizer.from_pretrained(name))

    # -- encode / decode -----------------------------------------------------

    def encode(
        self,
        sequence: str,
        pair: str | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        ids = self._tok.encode(sequence)
        tokens = [self._tok.id_to_token(i) or "" for i in ids]
        return Encoding(ids=ids, tokens=tokens)

    def encode_batch(
        self,
        input: list[str],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[Encoding]:
        all_ids = self._tok.encode_batch(input)
        result = []
        for ids in all_ids:
            tokens = [self._tok.id_to_token(i) or "" for i in ids]
            result.append(Encoding(ids=ids, tokens=tokens))
        return result

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tok.decode(ids)

    def decode_batch(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return self._tok.decode_batch(sequences)

    # -- vocab inspection ----------------------------------------------------

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return self._tok.vocab_size

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        return self._tok.id_to_token(id)
