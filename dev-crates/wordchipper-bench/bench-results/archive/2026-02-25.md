| Field        | Value                             |
| ------------ | --------------------------------- |
| **Date**     | `2026-02-25`                      |
| **Commit**   | `8faa0fd` (bpe-backtrack-encoder) |
| **Hardware** | Apple M4 Pro                      |

## Encoder Variants

- **buffer_sweep** - O(n^2) reference implementation using a separate working buffer
- **tail_sweep** - O(n^2) linear-scan BPE merge using the output buffer tail as working memory
- **merge_heap** - O(n^2) with parallel pair-rank tracking
- **priority_merge** - O(n log n) binary min-heap with doubly-linked list
- **bpe_backtrack** - O(n) Aho-Corasick greedy match with backtracking validation

## Single-String Encoding (median MB/s)

Corpus: `english.txt` (~7 KB) and `multilingual.txt` (~9 KB), repeated 10x.

### Without accelerated lexers (regex spanning)

| Encoder        | diverse cl100k | diverse o200k | english cl100k | english o200k |
| -------------- | -------------- | ------------- | -------------- | ------------- |
| buffer_sweep   | 16             | 11            | 15             | 10            |
| tail_sweep     | 16             | 11            | 15             | 10            |
| merge_heap     | 16             | 11            | 15             | 11            |
| priority_merge | 16             | 11            | 16             | 11            |
| bpe_backtrack  | 16             | 11            | 16             | 11            |
| bpe_openai     | 46             | 44            | 44             | 43            |
| tiktoken-rs    | 11             | 11            | 11             | 11            |
| HF tokenizers  | 7              | 7             | 6              | 6             |

### With accelerated lexers (logos spanning)

| Encoder        | diverse cl100k | diverse o200k | english cl100k | english o200k |
| -------------- | -------------- | ------------- | -------------- | ------------- |
| buffer_sweep   | 94             | 90            | 93             | 90            |
| tail_sweep     | 97             | 97            | 96             | 93            |
| merge_heap     | 101            | 97            | 100            | 96            |
| priority_merge | 128            | 121           | 126            | 120           |
| bpe_backtrack  | 101            | 97            | 100            | 97            |

## Parallel Batch Encoding (median MB/s)

Corpus: 1024 samples from fineweb-edu shard 0 (~4.2 MB batch). All engines use rayon `par_iter()`.

### Without accelerated lexers (regex spanning)

| Encoder        | cl100k | o200k |
| -------------- | ------ | ----- |
| buffer_sweep   | 264    | 154   |
| tail_sweep     | 261    | 137   |
| merge_heap     | 279    | 141   |
| priority_merge | 247    | 138   |
| bpe_backtrack  | 274    | 155   |

### With accelerated lexers (logos spanning)

| Encoder        | cl100k | o200k |
| -------------- | ------ | ----- |
| buffer_sweep   | 1,361  | 1,272 |
| tail_sweep     | 1,487  | 1,313 |
| merge_heap     | 1,284  | 1,148 |
| priority_merge | 1,193  | 1,070 |
| bpe_backtrack  | 1,310  | 1,032 |
| bpe_openai     | 115    | 93    |
| tiktoken-rs    | 153    | 142   |
| HF tokenizers  | 10     | 9     |
