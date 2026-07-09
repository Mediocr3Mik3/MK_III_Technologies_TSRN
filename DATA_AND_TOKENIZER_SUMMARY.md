# TropFormer Data & Tokenizer Summary

This document summarizes the tokenizers and pretraining token shards we have built for the TropFormer project on this machine.

## Tokenizers

| Tokenizer | Type | Vocab size | Location | Notes |
|-----------|------|------------|----------|-------|
| `gpt2_tsrn_base` | GPT-2 based (HuggingFace `GPT2TokenizerFast`) | 50,349 | `D:\ml\tsrn_data\tokenizer\gpt2_tsrn_base\` | Originally used for all local DirectML training. Extended with a small TSRN special-token set. |
| `tmt_48k.json` | Tropical Merging Tokenizer (TMT) | ~48,000 (planned) | `D:\ml\tsrn_data\tokenizer\tmt_48k.json` | Built for the cloud/Azure pipeline. Required by `create_shards.py`. Currently present on `D:` but **not** on `E:\ml\tsrn_data\tokenizer\`. |
| `tsrn_config.json` | Config metadata | — | `D:\ml\tsrn_data\tokenizer\tsrn_config.json` | Tokenizer configuration record. |

## Tokenized Shards

All pretraining shards live under:

```text
D:\ml\tsrn_data\shards\pretrain\
```

Each source subdirectory contains paired `.bin` / `.idx.npy` files. A `.done` marker indicates the source finished tokenization.

### Shard inventory

| Source | Shards | Size (MB) | Approx. tokens | `.done` | Notes |
|--------|--------|-----------|----------------|---------|-------|
| `arxiv_gist` | 9 | 6,800 | 1.78 B | ✅ | Added during June 2026 backfill |
| `cosmopedia` | 10 | 6,924 | 1.82 B | ✅ | Added during June 2026 backfill |
| `proofpile2` | 15 | 11,142 | 2.92 B | ✅ | Added during June 2026 backfill |
| `simple_wikipedia` | 1 | 269 | 70 M | ✅ | Added during June 2026 backfill |
| `stack_python` | 10 | 6,918 | 1.81 B | ✅ | Added during June 2026 backfill |
| `obo_ontology` | 1 | 105 | 28 M | ✅ | Added during June 2026 backfill (replaced the misnamed `ontologies` entry) |
| `finemath` | 43 | 32,542 | ~8.6 B | ✅ | Pre-existing local shard |
| `fineweb_edu` | 47 | 35,327 | ~9.4 B | ✅ | Pre-existing local shard |
| `openwebmath` | 68 | 51,554 | ~13.6 B | ✅ | Pre-existing local shard |
| `kyro` | 1 | 21 | ~5 M | ✅ | Proprietary/curated data |
| `nli` | 1 | 396 | ~104 M | ✅ | Curated reasoning data |
| `wordnet` | 1 | 15 | ~4 M | ✅ | Curated lexical data |
| `syn_type_a` | 1 | 288 | ~75 M | ✅ | Synthetic type-A data |
| `syn_type_b` | 1 | 334 | ~87 M | ✅ | Synthetic type-B data |
| `syn_type_c` | 2 | 974 | ~255 M | ✅ | Synthetic type-C data |
| `syn_type_d` | 1 | 147 | ~38 M | ✅ | Synthetic type-D data |
| `syn_type_e` | 1 | 300 | ~79 M | ✅ | Synthetic type-E data |
| `syn_type_f` | 1 | 402 | ~105 M | ✅ | Synthetic type-F data |
| `syn_type_g` | 1 | 408 | ~107 M | ✅ | Synthetic type-G data |
| `syn_type_h` | 1 | 336 | ~88 M | ✅ | Synthetic type-H data |

**Totals:** ~21 sources, ~214 shards, ~163 GB on disk, ~42 B tokens (estimated from the GPT-2-based 200 M-token shard size).

## Pipeline Scripts

| Script | Path | Purpose | Tokenizer used |
|--------|------|---------|----------------|
| `tokenize_shard_sequential.py` | `e:\ml\tsrn_data\pipeline\tokenize_shard_sequential.py` | Sequentially tokenizes each processed source into `.bin` / `.idx.npy` shards. | `gpt2_tsrn_base` |
| `create_shards.py` | `e:\ml\tsrn_data\pipeline\create_shards.py` | TMT-based shard creation with scoring/filtering and train/val split. | `tmt_48k.json` |
| `extend_tokenizer.py` | `e:\ml\tsrn_data\pipeline\extend_tokenizer.py` | Trains or extends the Tropical Merging Tokenizer to 48 k vocab. | Output: `tmt_48k.json` |
| `prepare_corpus.py` | `e:\ml\tsrn_data\pipeline\prepare_corpus.py` | Formats raw data into unified records in `processed/`. | None (pre-tokenization) |

## Key History

1. **Initial local shards** were produced with the GPT-2 tokenizer (`gpt2_tsrn_base`) and covered `finemath`, `fineweb_edu`, `openwebmath`, `kyro`, `nli`, `wordnet`, and the eight `syn_type_*` sources.
2. **June 2026 backfill:** we identified six missing curated sources from the Azure pretrain manifest — `arxiv_gist`, `cosmopedia`, `proofpile2`, `simple_wikipedia`, `stack_python`, and `obo_ontology`.
3. We updated the `SOURCE_ORDER` list in `tokenize_shard_sequential.py` to include all processed sources and ran it with `--resume` so it skipped already-complete sources and filled in the missing ones.
4. The backfill completed in ~4.5 hours, adding ~47 new shards and ~8.4 B tokens.
5. The `tmt_48k.json` tokenizer was later built for the Azure/cloud pipeline; `create_shards.py` will use it if we re-tokenize the raw corpus for cloud training.

## Current State

- The full GPT-2-based shard set is ready on `D:` for local DirectML training.
- The `tmt_48k.json` tokenizer is available on `D:` but **not yet mirrored** to `E:\ml\tsrn_data\tokenizer\`.
- No further cloud upload or re-tokenization has been performed yet.

## Locations Summary

```text
D:\ml\tsrn_data\tokenizer\gpt2_tsrn_base   # GPT-2 tokenizer used locally
D:\ml\tsrn_data\tokenizer\tmt_48k.json     # TMT tokenizer for cloud pipeline
D:\ml\tsrn_data\shards\pretrain\            # All pretraining shards
E:\ml\tsrn_data\                            # Intended cloud/raw data root (currently lacks tokenizer mirror)
```
