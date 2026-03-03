import os
import tempfile

import pytest
from wordchipper import Tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.from_pretrained("cl100k_base")


# Construction


def test_from_pretrained_invalid():
    with pytest.raises(ValueError):
        Tokenizer.from_pretrained("nonexistent_model")


def test_from_pretrained_empty_name():
    with pytest.raises(ValueError):
        Tokenizer.from_pretrained("")


PRETRAINED_MODELS = [
    "openai:r50k_base",
    "openai:p50k_base",
    "openai:p50k_edit",
    "openai:cl100k_base",
    "openai:o200k_base",
    "openai:o200k_harmony",
]


@pytest.mark.parametrize("model", PRETRAINED_MODELS)
def test_from_pretrained_all_models(model):
    tok = Tokenizer.from_pretrained(model)
    tokens = tok.encode("hello")
    assert len(tokens) > 0
    assert tok.decode(tokens) == "hello"


# Encode


def test_encode(tokenizer):
    tokens = tokenizer.encode("hello world")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0


def test_encode_empty(tokenizer):
    assert tokenizer.encode("") == []


def test_encode_whitespace(tokenizer):
    tokens = tokenizer.encode("   ")
    assert len(tokens) > 0


def test_encode_newlines(tokenizer):
    tokens = tokenizer.encode("line1\nline2\nline3")
    assert len(tokens) > 0


def test_encode_unicode(tokenizer):
    texts = [
        "Hello, world!",
        "Bonjour le monde",
        "Hallo Welt",
        "Hola mundo",
    ]
    for text in texts:
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0
        assert tokenizer.decode(tokens) == text


def test_encode_cjk(tokenizer):
    text = "\u4f60\u597d\u4e16\u754c"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    assert tokenizer.decode(tokens) == text


def test_encode_emoji(tokenizer):
    text = "hello \U0001f600 world \U0001f680"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    assert tokenizer.decode(tokens) == text


def test_encode_special_characters(tokenizer):
    text = "a\tb\nc\r\nd"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text


def test_encode_long_text(tokenizer):
    text = "word " * 1000
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    assert tokenizer.decode(tokens) == text


def test_encode_single_character(tokenizer):
    for ch in "abcABC012!@#":
        tokens = tokenizer.encode(ch)
        assert len(tokens) > 0
        assert tokenizer.decode(tokens) == ch


def test_encode_repeated_text(tokenizer):
    text = "abc" * 100
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text


# Decode


def test_decode_roundtrip(tokenizer):
    text = "hello world"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_decode_empty(tokenizer):
    assert tokenizer.decode([]) == ""


def test_decode_single_token(tokenizer):
    tokens = tokenizer.encode("hello")
    for t in tokens:
        result = tokenizer.decode([t])
        assert isinstance(result, str)
        assert len(result) > 0


# Encode Batch


def test_encode_batch(tokenizer):
    texts = ["hello", "world", "foo bar"]
    results = tokenizer.encode_batch(texts)
    assert len(results) == 3
    for tokens in results:
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)


def test_encode_batch_empty(tokenizer):
    assert tokenizer.encode_batch([]) == []


def test_encode_batch_single_item(tokenizer):
    results = tokenizer.encode_batch(["hello"])
    assert len(results) == 1


def test_encode_batch_with_empty_strings(tokenizer):
    results = tokenizer.encode_batch(["hello", "", "world"])
    assert len(results) == 3
    assert results[1] == []


def test_encode_batch_large(tokenizer):
    texts = [f"sentence number {i}" for i in range(100)]
    results = tokenizer.encode_batch(texts)
    assert len(results) == 100
    decoded = tokenizer.decode_batch(results)
    assert decoded == texts


# Decode Batch


def test_decode_batch_roundtrip(tokenizer):
    texts = ["hello", "world", "foo bar"]
    batch = tokenizer.encode_batch(texts)
    decoded = tokenizer.decode_batch(batch)
    assert decoded == texts


def test_decode_batch_empty(tokenizer):
    assert tokenizer.decode_batch([]) == []


def test_decode_batch_with_empty_token_lists(tokenizer):
    results = tokenizer.decode_batch([[], []])
    assert results == ["", ""]


# Consistency


def test_encode_deterministic(tokenizer):
    text = "The quick brown fox jumps over the lazy dog."
    tokens1 = tokenizer.encode(text)
    tokens2 = tokenizer.encode(text)
    assert tokens1 == tokens2


def test_encode_batch_matches_individual(tokenizer):
    texts = ["hello world", "foo bar baz", "testing 123"]
    batch_results = tokenizer.encode_batch(texts)
    individual_results = [tokenizer.encode(t) for t in texts]
    assert batch_results == individual_results


def test_decode_batch_matches_individual(tokenizer):
    texts = ["hello world", "foo bar baz", "testing 123"]
    batch_tokens = tokenizer.encode_batch(texts)
    batch_decoded = tokenizer.decode_batch(batch_tokens)
    individual_decoded = [tokenizer.decode(t) for t in batch_tokens]
    assert batch_decoded == individual_decoded


# Vocab Inspection


def test_vocab_size(tokenizer):
    size = tokenizer.vocab_size
    assert isinstance(size, int)
    assert size > 0


def test_max_token(tokenizer):
    max_token = tokenizer.max_token
    assert max_token is not None
    assert isinstance(max_token, int)
    assert max_token > 0


def test_vocab_size_known_value():
    tok = Tokenizer.from_pretrained("cl100k_base")
    assert tok.vocab_size == 100256


def test_token_to_id(tokenizer):
    token_id = tokenizer.token_to_id("hello")
    assert isinstance(token_id, int)


def test_token_to_id_unknown(tokenizer):
    result = tokenizer.token_to_id("xyzzy_not_a_real_token_99999")
    assert result is None


def test_id_to_token(tokenizer):
    token_id = tokenizer.token_to_id("hello")
    assert token_id is not None
    token_str = tokenizer.id_to_token(token_id)
    assert token_str == "hello"


def test_id_to_token_unknown(tokenizer):
    result = tokenizer.id_to_token(999_999_999)
    assert result is None


def test_token_to_id_id_to_token_roundtrip(tokenizer):
    for word in ["the", " the", "hello", " world", "abc"]:
        token_id = tokenizer.token_to_id(word)
        if token_id is not None:
            assert tokenizer.id_to_token(token_id) == word


def test_token_to_id_single_bytes(tokenizer):
    for byte_val in [ord("a"), ord("z"), ord("0"), ord(" ")]:
        token_str = chr(byte_val)
        token_id = tokenizer.token_to_id(token_str)
        assert token_id is not None


# Special Tokens


def test_get_special_tokens(tokenizer):
    specials = tokenizer.get_special_tokens()
    assert isinstance(specials, list)
    assert len(specials) > 0
    for name, token_id in specials:
        assert isinstance(name, str)
        assert isinstance(token_id, int)


def test_special_tokens_contain_endoftext():
    tok = Tokenizer.from_pretrained("cl100k_base")
    specials = tok.get_special_tokens()
    names = [name for name, _ in specials]
    assert "<|endoftext|>" in names


# Available Models


def test_available_models():
    models = Tokenizer.available_models()
    assert isinstance(models, list)
    assert len(models) > 0
    for name in ["openai:r50k_base", "openai:p50k_base", "openai:cl100k_base", "openai:o200k_base"]:
        assert name in models


# Save Vocab


def test_save_base64_vocab(tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "vocab.tiktoken")
        tokenizer.save_base64_vocab(path)
        assert os.path.isfile(path)
        with open(path) as f:
            line_count = sum(1 for _ in f)
        # span_vocab excludes special tokens and byte-level tokens
        assert 0 < line_count <= tokenizer.vocab_size
