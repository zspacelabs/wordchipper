| Field        | Value                                              |
| ------------ | -------------------------------------------------- |
| **Date**     | `2026-03-03`                                       |
| **Commit**   | `bc69229` (issue-266-equivalence-class-tests)      |
| **Hardware** | Apple M4 Pro                                       |

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
| buffer_sweep   | 15             | 10            | 16             | 11            |
| tail_sweep     | 16             | 10            | 16             | 11            |
| merge_heap     | 16             | 10            | 16             | 11            |
| priority_merge | 17             | 12            | 16             | 11            |
| bpe_backtrack  | 16             | 12            | 16             | 11            |
| bpe_openai     | 43             | 43            | 47             | 46            |
| tiktoken-rs    | 11             | 11            | 12             | 11            |
| HF tokenizers  | 7              | 7             | 7              | 7             |

### With accelerated lexers (logos spanning)

| Encoder        | diverse cl100k | diverse o200k | english cl100k | english o200k |
| -------------- | -------------- | ------------- | -------------- | ------------- |
| buffer_sweep   | 61             | 31            | 86             | 81            |
| tail_sweep     | 56             | 30            | 81             | 77            |
| merge_heap     | 61             | 38            | 91             | 86            |
| priority_merge | 88             | 81            | 110            | 107           |
| bpe_backtrack  | 79             | 80            | 108            | 104           |

## Parallel Batch Encoding (median MB/s)

Corpus: 1024 samples from fineweb-edu shard 0 (~4.2 MB batch). All engines use rayon `par_iter()`.

### Without accelerated lexers (regex spanning)

| Encoder        | cl100k | o200k |
| -------------- | ------ | ----- |
| buffer_sweep   | 276    | 137   |
| tail_sweep     | 260    | 135   |
| merge_heap     | 264    | 135   |
| priority_merge | 243    | 130   |
| bpe_backtrack  | 278    | 141   |

### With accelerated lexers (logos spanning)

| Encoder        | cl100k | o200k |
| -------------- | ------ | ----- |
| buffer_sweep   | 1,315  | 1,174 |
| tail_sweep     | 1,293  | 1,179 |
| merge_heap     | 1,308  | 1,182 |
| priority_merge | 880    | 832   |
| bpe_backtrack  | 1,114  | 1,112 |
| bpe_openai     | 96     | 119   |
| tiktoken-rs    | 162    | 142   |
| HF tokenizers  | 10     | 9     |
