"""Azure cloud training pipeline for TropFormer / Kyro.

Three stages:
  1. Pretraining (92B tokens, 11 datasets)
  2. SFT (~6.5M examples, 3 epochs, curriculum: reasoning -> instruction -> tool -> kyro)
  3. DPO (25K preference pairs)

All stages use the Tropical Merging Tokenizer (TMT).

See ``research/cloud/azure/README.md`` for the full pipeline.
"""
