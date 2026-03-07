"""Python encode/encode_batch benchmarks: wordchipper vs tiktoken vs tokenizers.

Matches encoding_single.rs and encoding_parallel.rs:
  - Single-string: english.txt / multilingual.txt repeated 10x
  - Batch: 1024 samples from fineweb-edu shard 0 (~4.2 MB)

Build the extension in release mode first for meaningful numbers:
    maturin develop --release

Run with:
    pytest benchmarks/
"""

import pytest

import wordchipper

MODELS = ["gpt2", "r50k_base", "cl100k_base", "o200k_base"]

# HuggingFace model identifiers (matching the Rust benchmarks)
HF_MODELS = {
    "gpt2": "Xenova/gpt2",
    "r50k_base": "Xenova/gpt-3",
    "cl100k_base": "Xenova/text-embedding-ada-002",
    "o200k_base": "Xenova/gpt-4o",
}


def _utf8_len(text):
    return len(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Single-string encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
class TestSingleEncode:
    def test_wordchipper_english(self, benchmark, model, english_text):
        options = wordchipper.TokenizerOptions.default()
        options.set_parallel(False)
        options.set_accelerated_lexers(False)

        tok = wordchipper.Tokenizer.from_pretrained(model, options)
        benchmark.group = f"single/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.encode, english_text)

    def test_wordchipper_english_accel(self, benchmark, model, english_text):
        options = wordchipper.TokenizerOptions.default()
        options.set_parallel(False)
        options.set_accelerated_lexers(True)

        tok = wordchipper.Tokenizer.from_pretrained(model, options)
        benchmark.group = f"single/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.encode, english_text)

    def test_wordchipper_diverse(self, benchmark, model, diverse_text):
        options = wordchipper.TokenizerOptions.default()
        options.set_parallel(False)
        options.set_accelerated_lexers(False)

        tok = wordchipper.Tokenizer.from_pretrained(model, options)
        benchmark.group = f"single/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.encode, diverse_text)

    def test_wordchipper_diverse_accel(self, benchmark, model, diverse_text):
        options = wordchipper.TokenizerOptions.default()
        options.set_parallel(False)
        options.set_accelerated_lexers(True)

        tok = wordchipper.Tokenizer.from_pretrained(model, options)
        benchmark.group = f"single/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.encode, diverse_text)

    def test_tiktoken_english(self, benchmark, model, english_text):
        import tiktoken

        tok = tiktoken.get_encoding(model)
        benchmark.group = f"single/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.encode, english_text, allowed_special="all")

    def test_tiktoken_diverse(self, benchmark, model, diverse_text):
        import tiktoken

        tok = tiktoken.get_encoding(model)
        benchmark.group = f"single/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.encode, diverse_text, allowed_special="all")

    def test_tokenizers_english(self, benchmark, model, english_text):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        benchmark.group = f"single/english/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(english_text)
        benchmark(tok.encode, english_text)

    def test_tokenizers_diverse(self, benchmark, model, diverse_text):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        benchmark.group = f"single/diverse/{model}"
        benchmark.extra_info["input_bytes"] = _utf8_len(diverse_text)
        benchmark(tok.encode, diverse_text)


# ---------------------------------------------------------------------------
# Parallel batch encoding (1024 samples from fineweb-edu)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
class TestBatchEncode:
    def test_wordchipper_parallel_accel(self, benchmark, model, fineweb_batch):
        texts, total_bytes = fineweb_batch

        options = wordchipper.TokenizerOptions.default()
        options.set_parallel(True)
        options.set_accelerated_lexers(True)

        tok = wordchipper.Tokenizer.from_pretrained(model, options)
        benchmark.group = f"batch/{model}"
        benchmark.extra_info["input_bytes"] = total_bytes
        benchmark(tok.encode_batch, texts)

    def test_wordchipper_parallel(self, benchmark, model, fineweb_batch):
        texts, total_bytes = fineweb_batch

        options = wordchipper.TokenizerOptions.default()
        options.set_parallel(True)
        options.set_accelerated_lexers(False)

        tok = wordchipper.Tokenizer.from_pretrained(model, options)
        benchmark.group = f"batch/{model}"
        benchmark.extra_info["input_bytes"] = total_bytes
        benchmark(tok.encode_batch, texts)

    def test_wordchipper_threadpool(self, benchmark, model, fineweb_batch, max_threads):
        texts, total_bytes = fineweb_batch

        from concurrent.futures import ThreadPoolExecutor

        options = wordchipper.TokenizerOptions.default()
        options.set_parallel(False)
        options.set_concurrent(True)
        options.set_accelerated_lexers(False)

        tok = wordchipper.Tokenizer.from_pretrained(model, options)
        benchmark.group = f"batch/{model}"
        benchmark.extra_info["input_bytes"] = total_bytes

        pool = ThreadPoolExecutor(max_workers=max_threads)

        def encode_batch_threaded(texts):
            return list(pool.map(tok.encode, texts))

        benchmark(encode_batch_threaded, texts)
        pool.shutdown(wait=False)

    def test_wordchipper_threadpool_accel(self, benchmark, model, fineweb_batch, max_threads):
        texts, total_bytes = fineweb_batch

        from concurrent.futures import ThreadPoolExecutor

        options = wordchipper.TokenizerOptions.default()
        options.set_parallel(False)
        options.set_concurrent(True)
        options.set_accelerated_lexers(True)

        tok = wordchipper.Tokenizer.from_pretrained(model, options)
        benchmark.group = f"batch/{model}"
        benchmark.extra_info["input_bytes"] = total_bytes

        pool = ThreadPoolExecutor(max_workers=max_threads)

        def encode_batch_threaded(texts):
            return list(pool.map(tok.encode, texts))

        benchmark(encode_batch_threaded, texts)
        pool.shutdown(wait=False)

    def test_tiktoken(self, benchmark, model, fineweb_batch):
        texts, total_bytes = fineweb_batch

        import tiktoken

        tok = tiktoken.get_encoding(model)
        benchmark.group = f"batch/{model}"
        benchmark.extra_info["input_bytes"] = total_bytes
        benchmark(tok.encode_batch, texts, allowed_special="all")

    def test_tokenizers(self, benchmark, model, fineweb_batch):
        texts, total_bytes = fineweb_batch

        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(HF_MODELS[model])
        benchmark.group = f"batch/{model}"
        benchmark.extra_info["input_bytes"] = total_bytes
        benchmark(tok.encode_batch, texts)
