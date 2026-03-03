"""Tests for wordchipper.compat (tiktoken and tokenizers compatibility layers)."""

import pytest
from wordchipper import Tokenizer
from wordchipper.compat import tiktoken
from wordchipper.compat.tokenizers import Encoding as HFEncoding
from wordchipper.compat.tokenizers import Tokenizer as HFTokenizer


# ===================================================================
# tiktoken compat
# ===================================================================


class TestTiktokenGetEncoding:
    def test_get_encoding(self):
        enc = tiktoken.get_encoding("cl100k_base")
        assert enc.name == "cl100k_base"

    def test_get_encoding_cached(self):
        enc1 = tiktoken.get_encoding("cl100k_base")
        enc2 = tiktoken.get_encoding("cl100k_base")
        assert enc1 is enc2

    def test_get_encoding_unknown(self):
        with pytest.raises(ValueError, match="Unknown encoding"):
            tiktoken.get_encoding("nonexistent")

    def test_list_encoding_names(self):
        names = tiktoken.list_encoding_names()
        assert "cl100k_base" in names
        assert "o200k_base" in names
        assert "r50k_base" in names


class TestTiktokenModelMapping:
    def test_encoding_for_model(self):
        enc = tiktoken.encoding_for_model("gpt-4o")
        assert enc.name == "o200k_base"

    def test_encoding_for_model_chat(self):
        enc = tiktoken.encoding_for_model("gpt-4")
        assert enc.name == "cl100k_base"

    def test_encoding_name_for_model(self):
        assert tiktoken.encoding_name_for_model("gpt-4o") == "o200k_base"
        assert tiktoken.encoding_name_for_model("gpt-4") == "cl100k_base"
        assert tiktoken.encoding_name_for_model("davinci") == "r50k_base"

    def test_encoding_name_for_model_prefix(self):
        assert tiktoken.encoding_name_for_model("gpt-4o-2024-08-06") == "o200k_base"
        assert tiktoken.encoding_name_for_model("gpt-4-0613") == "cl100k_base"

    def test_encoding_name_for_model_finetuned(self):
        assert tiktoken.encoding_name_for_model("ft:gpt-4o:my-org") == "o200k_base"

    def test_encoding_for_model_prefix(self):
        enc = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        assert enc.name == "o200k_base"

    def test_encoding_name_for_model_unknown(self):
        with pytest.raises(KeyError):
            tiktoken.encoding_name_for_model("totally-unknown-model")


class TestTiktokenEncoding:
    @pytest.fixture(scope="class")
    def enc(self):
        return tiktoken.get_encoding("cl100k_base")

    def test_encode_decode_roundtrip(self, enc):
        text = "hello world"
        tokens = enc.encode(text)
        assert isinstance(tokens, list)
        assert enc.decode(tokens) == text

    def test_encode_ordinary(self, enc):
        text = "hello world"
        assert enc.encode_ordinary(text) == enc.encode(text)

    def test_encode_batch(self, enc):
        texts = ["hello", "world"]
        results = enc.encode_batch(texts)
        assert len(results) == 2
        for i, text in enumerate(texts):
            assert enc.decode(results[i]) == text

    def test_encode_ordinary_batch(self, enc):
        texts = ["hello", "world"]
        assert enc.encode_ordinary_batch(texts) == enc.encode_batch(texts)

    def test_decode_batch(self, enc):
        texts = ["hello", "world"]
        batch = enc.encode_batch(texts)
        assert enc.decode_batch(batch) == texts

    def test_encode_empty(self, enc):
        assert enc.encode("") == []

    def test_decode_empty(self, enc):
        assert enc.decode([]) == ""

    def test_encode_accepts_special_kwargs(self, enc):
        # These kwargs are accepted silently for API compat
        tokens = enc.encode("hello", allowed_special="all", disallowed_special=())
        assert enc.decode(tokens) == "hello"


class TestTiktokenProperties:
    @pytest.fixture(scope="class")
    def enc(self):
        return tiktoken.get_encoding("cl100k_base")

    def test_n_vocab(self, enc):
        assert enc.n_vocab > 0

    def test_max_token_value(self, enc):
        assert enc.max_token_value > 0

    def test_eot_token(self, enc):
        assert isinstance(enc.eot_token, int)

    def test_special_tokens_set(self, enc):
        specials = enc.special_tokens_set
        assert isinstance(specials, set)
        assert "<|endoftext|>" in specials


class TestTiktokenMatchesWordchipper:
    def test_encode_matches(self):
        enc = tiktoken.get_encoding("cl100k_base")
        tok = Tokenizer.from_pretrained("cl100k_base")
        text = "The quick brown fox jumps over the lazy dog."
        assert enc.encode(text) == tok.encode(text)

    def test_decode_matches(self):
        enc = tiktoken.get_encoding("cl100k_base")
        tok = Tokenizer.from_pretrained("cl100k_base")
        tokens = tok.encode("hello world")
        assert enc.decode(tokens) == tok.decode(tokens)


# ===================================================================
# tokenizers compat
# ===================================================================


class TestHFEncoding:
    def test_encoding_dataclass(self):
        enc = HFEncoding(ids=[1, 2, 3], tokens=["a", "b", "c"])
        assert enc.ids == [1, 2, 3]
        assert enc.tokens == ["a", "b", "c"]


class TestHFTokenizer:
    @pytest.fixture(scope="class")
    def tok(self):
        return HFTokenizer.from_pretrained("Xenova/gpt-4o")

    def test_from_pretrained_hf_id(self):
        tok = HFTokenizer.from_pretrained("Xenova/gpt-4o")
        result = tok.encode("hello")
        assert len(result.ids) > 0

    def test_from_pretrained_bare_name(self):
        tok = HFTokenizer.from_pretrained("cl100k_base")
        result = tok.encode("hello")
        assert len(result.ids) > 0

    def test_from_pretrained_unknown(self):
        with pytest.raises(ValueError):
            HFTokenizer.from_pretrained("Xenova/totally-unknown")

    def test_encode_returns_encoding(self, tok):
        result = tok.encode("hello world")
        assert isinstance(result, HFEncoding)
        assert isinstance(result.ids, list)
        assert isinstance(result.tokens, list)
        assert len(result.ids) == len(result.tokens)

    def test_encode_decode_roundtrip(self, tok):
        text = "hello world"
        enc = tok.encode(text)
        assert tok.decode(enc.ids) == text

    def test_encode_batch(self, tok):
        texts = ["hello", "world"]
        results = tok.encode_batch(texts)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, HFEncoding)
            assert len(r.ids) == len(r.tokens)

    def test_decode_batch(self, tok):
        texts = ["hello", "world"]
        batch = tok.encode_batch(texts)
        decoded = tok.decode_batch([r.ids for r in batch])
        assert decoded == texts

    def test_get_vocab_size(self, tok):
        size = tok.get_vocab_size()
        assert isinstance(size, int)
        assert size > 0

    def test_token_to_id(self, tok):
        tid = tok.token_to_id("hello")
        assert isinstance(tid, int)

    def test_token_to_id_unknown(self, tok):
        assert tok.token_to_id("xyzzy_not_real_99999") is None

    def test_id_to_token(self, tok):
        tid = tok.token_to_id("hello")
        assert tid is not None
        assert tok.id_to_token(tid) == "hello"

    def test_id_to_token_out_of_range(self, tok):
        assert tok.id_to_token(999_999_999) is None

    def test_encode_empty(self, tok):
        result = tok.encode("")
        assert result.ids == []
        assert result.tokens == []

    def test_decode_empty(self, tok):
        assert tok.decode([]) == ""

    def test_encode_accepts_extra_kwargs(self, tok):
        # Extra params accepted silently for API compat
        result = tok.encode("hello", pair=None, is_pretokenized=False)
        assert len(result.ids) > 0


class TestHFTokenizerMatchesWordchipper:
    def test_encode_ids_match(self):
        hf = HFTokenizer.from_pretrained("cl100k_base")
        wc = Tokenizer.from_pretrained("cl100k_base")
        text = "The quick brown fox jumps over the lazy dog."
        assert hf.encode(text).ids == wc.encode(text)

    def test_vocab_size_matches(self):
        hf = HFTokenizer.from_pretrained("cl100k_base")
        wc = Tokenizer.from_pretrained("cl100k_base")
        assert hf.get_vocab_size() == wc.vocab_size
