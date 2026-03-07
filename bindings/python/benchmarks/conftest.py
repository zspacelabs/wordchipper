import os
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "dev-crates" / "wordchipper-bench" / "benches" / "data"

SHARD_URL = (
    "https://huggingface.co/datasets/karpathy/"
    "fineweb-edu-100b-shuffle/resolve/main/shard_00000.parquet"
)
SHARD_CACHE = Path("~/.cache/brn-nanochat/dataset/shard_00000.parquet").expanduser()
BATCH_SIZE = 1024


@pytest.fixture(scope="session")
def max_threads():
    """Max thread count from RAYON_NUM_THREADS, or None (use system default)."""
    val = os.environ.get("RAYON_NUM_THREADS")
    if not val:
        return None
    n = int(val)
    if n < 1:
        raise pytest.UsageError(f"RAYON_NUM_THREADS must be >= 1, got {val!r}")
    return n


@pytest.fixture(scope="session")
def english_text():
    return (DATA_DIR / "english.txt").read_text(encoding="utf-8") * 10


@pytest.fixture(scope="session")
def diverse_text():
    return (DATA_DIR / "multilingual.txt").read_text(encoding="utf-8") * 10


@pytest.fixture(scope="session")
def fineweb_batch():
    """Load 1024 text samples from fineweb-edu shard 0, matching the Rust bench."""
    import pyarrow.parquet as pq

    if not SHARD_CACHE.exists():
        SHARD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SHARD_URL, SHARD_CACHE)

    pf = pq.ParquetFile(SHARD_CACHE)
    texts = []
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["text"]):
        texts = batch.column("text").to_pylist()
        break
    total_bytes = sum(len(s.encode("utf-8")) for s in texts)
    return texts, total_bytes


def pytest_configure(config):
    if hasattr(config.option, "benchmark_columns"):
        config.option.benchmark_columns = ["median"]
        config.option.benchmark_sort = "mean"
        config.option.benchmark_group_by = "group"


def pytest_terminal_summary(terminalreporter, config):
    try:
        benchmarks = config._benchmarksession.benchmarks
    except AttributeError:
        return
    if not benchmarks:
        return

    rows = []
    for bench in benchmarks:
        input_bytes = bench.extra_info.get("input_bytes")
        if not input_bytes or not bench.stats:
            continue
        mb_s = input_bytes / bench.stats.median / 1_000_000
        rows.append((bench.group or "", bench.name, mb_s))

    if not rows:
        return

    terminalreporter.section("throughput (median MB/s)", sep="-")
    current_group = None
    for group, name, mb_s in sorted(rows):
        if group != current_group:
            current_group = group
            terminalreporter.write_line(f"\n  {group}:")
        terminalreporter.write_line(f"    {name:55s} {mb_s:>10,.1f} MB/s")
