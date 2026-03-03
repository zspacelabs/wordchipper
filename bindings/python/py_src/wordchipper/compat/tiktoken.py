"""Drop-in replacement for the ``tiktoken`` library, backed by wordchipper.

Typical migration::

    # Before
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # After
    from wordchipper.compat import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
"""

from __future__ import annotations

from typing import Any

from wordchipper import Tokenizer

# ---------------------------------------------------------------------------
# Model-to-encoding mappings (from tiktoken 0.12, excluding gpt-2 entries)
# ---------------------------------------------------------------------------

MODEL_TO_ENCODING: dict[str, str] = {
    # chat
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    # reasoning
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    # base
    "davinci-002": "cl100k_base",
    "babbage-002": "cl100k_base",
    # embeddings
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # DALL-E
    "dall-e-2": "cl100k_base",
    "dall-e-3": "cl100k_base",
    # code
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "davinci-codex": "p50k_base",
    "cushman-codex": "p50k_base",
    # edit
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    # old completions
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    # old embeddings
    "text-similarity-davinci-001": "r50k_base",
    "text-similarity-curie-001": "r50k_base",
    "text-similarity-babbage-001": "r50k_base",
    "text-similarity-ada-001": "r50k_base",
    "text-search-davinci-doc-001": "r50k_base",
    "text-search-curie-doc-001": "r50k_base",
    "text-search-babbage-doc-001": "r50k_base",
    "text-search-ada-doc-001": "r50k_base",
    "code-search-babbage-code-001": "r50k_base",
    "code-search-ada-code-001": "r50k_base",
}

MODEL_PREFIX_TO_ENCODING: dict[str, str] = {
    "gpt-4o-": "o200k_base",
    "gpt-4-": "cl100k_base",
    "gpt-3.5-turbo-": "cl100k_base",
    "ft:gpt-4o": "o200k_base",
    "ft:gpt-4": "cl100k_base",
    "ft:gpt-3.5-turbo": "cl100k_base",
    "ft:davinci-002": "cl100k_base",
    "ft:babbage-002": "cl100k_base",
}

_ENCODING_NAMES = ["r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base"]

# Thread-safe encoding cache (keyed by encoding name)
_cache: dict[str, Encoding] = {}


class Encoding:
    """Wrapper around :class:`wordchipper.Tokenizer` with tiktoken's API."""

    def __init__(self, name: str, tokenizer: Tokenizer) -> None:
        self._name = name
        self._tok = tokenizer

    # -- properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_token_value(self) -> int:
        val = self._tok.max_token
        return val if val is not None else 0

    @property
    def n_vocab(self) -> int:
        return self.max_token_value + 1

    @property
    def eot_token(self) -> int:
        for tok_name, tok_id in self._tok.get_special_tokens():
            if tok_name == "<|endoftext|>":
                return tok_id
        return self.max_token_value

    @property
    def special_tokens_set(self) -> set[str]:
        return {name for name, _ in self._tok.get_special_tokens()}

    # -- encode / decode -----------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        allowed_special: Any = None,
        disallowed_special: Any = None,
    ) -> list[int]:
        return self._tok.encode(text)

    def encode_ordinary(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def encode_batch(
        self,
        text: list[str],
        *,
        allowed_special: Any = None,
        disallowed_special: Any = None,
    ) -> list[list[int]]:
        return self._tok.encode_batch(text)

    def encode_ordinary_batch(self, text: list[str]) -> list[list[int]]:
        return self._tok.encode_batch(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tok.decode(tokens)

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        return self._tok.decode_batch(batch)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def get_encoding(encoding_name: str) -> Encoding:
    """Return an :class:`Encoding` for the given encoding name (cached)."""
    if encoding_name not in _ENCODING_NAMES:
        raise ValueError(
            f"Unknown encoding {encoding_name!r}. "
            f"Available: {', '.join(_ENCODING_NAMES)}"
        )
    if encoding_name not in _cache:
        tok = Tokenizer.from_pretrained(encoding_name)
        _cache[encoding_name] = Encoding(encoding_name, tok)
    return _cache[encoding_name]


def encoding_name_for_model(model_name: str) -> str:
    """Return the encoding name for a model (without loading the encoding)."""
    if model_name in MODEL_TO_ENCODING:
        return MODEL_TO_ENCODING[model_name]
    for prefix, enc_name in MODEL_PREFIX_TO_ENCODING.items():
        if model_name.startswith(prefix):
            return enc_name
    raise KeyError(f"No encoding for model {model_name!r}")


def encoding_for_model(model_name: str) -> Encoding:
    """Return an :class:`Encoding` for the given model name."""
    return get_encoding(encoding_name_for_model(model_name))


def list_encoding_names() -> list[str]:
    """Return the list of available encoding names."""
    return list(_ENCODING_NAMES)
