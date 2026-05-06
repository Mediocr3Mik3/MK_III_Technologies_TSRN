# TropFormer Azure Training Pipeline

End-to-end pipeline to train TropFormer/Kyro on Azure: **pretrain (92B tokens) → SFT (~6.5M examples, curriculum) → DPO (25K preference pairs)**, all using the **Tropical Merging Tokenizer (TMT)**.

```
research/cloud/azure/
├── README.md                  ← you are here
├── data/
│   ├── manifests/
│   │   ├── pretrain_mix.yaml  ← 11 datasets, 92B tokens, weighted
│   │   ├── sft_mix.yaml       ← 21 datasets, curriculum, ~6.5M examples
│   │   └── dpo_mix.yaml       ← UltraFeedback + Kyro-specific, 25K pairs
│   ├── download.py            ← stream HF → blob shards (.jsonl.zst)
│   ├── train_tmt.py           ← train Tropical Merging Tokenizer (32k vocab)
│   ├── tokenize_shard.py      ← tokenize raw → packed .bin/.idx.npy
│   └── streaming_dataset.py   ← weighted iterable dataset for training
├── configs/
│   ├── pretrain_h100x8.py     ← Pro tier, 150M params, ctx 2048
│   ├── sft_h100x8.py          ← ctx 4096, curriculum config
│   └── dpo_h100x8.py          ← ctx 4096, beta=0.1
├── jobs/                      ← Azure ML job specs (primary path)
│   ├── environment.yaml       ← AzureML curated env
│   ├── environment.conda.yaml
│   ├── aml_data_download.yaml
│   ├── aml_train_tmt.yaml
│   ├── aml_tokenize_shard.yaml
│   ├── aml_pretrain.yaml
│   ├── aml_sft.yaml
│   └── aml_dpo.yaml
├── scripts/                   ← Azure VM scripts (fallback / iteration)
│   ├── azure_provision.sh     ← create VM + storage account
│   ├── blob_mount.sh          ← mount blob container at /mnt/blob
│   ├── run_pretrain.sh        ← VM pretrain launcher (download → tokenize → train)
│   ├── run_sft.sh             ← VM SFT launcher
│   ├── run_dpo.sh             ← VM DPO launcher
│   └── run_full_pipeline.sh   ← all three in sequence
├── train_pretrain_cloud.py    ← pretrain trainer (DDP-aware)
├── train_sft_cloud.py         ← curriculum SFT trainer
└── train_dpo_cloud.py         ← DPO trainer
```

## Path A — Azure ML (recommended)

```bash
# 0. one-time: create AML workspace + cluster + datastore (manual via portal or `az ml`)
az login
az ml environment create -f research/cloud/azure/jobs/environment.yaml

# 1. download raw data → blob (CPU job)
az ml job create -f research/cloud/azure/jobs/aml_data_download.yaml

# 2. train TMT tokenizer (1x GPU job, ~1h)
az ml job create -f research/cloud/azure/jobs/aml_train_tmt.yaml

# 3. tokenize + shard (CPU cluster job)
az ml job create -f research/cloud/azure/jobs/aml_tokenize_shard.yaml

# 4. pretrain (8x H100 job)
az ml job create -f research/cloud/azure/jobs/aml_pretrain.yaml

# 5. SFT
az ml job create -f research/cloud/azure/jobs/aml_sft.yaml

# 6. DPO
az ml job create -f research/cloud/azure/jobs/aml_dpo.yaml
```

> **Edit `compute:` in each YAML** to match your AML compute target names (e.g. `h100-cluster`, `cpu-cluster`).
> **Edit `path:` in datastore inputs/outputs** to match your blob datastore name.
> Set `WANDB_API_KEY` as a workspace secret to enable WandB logging.

## Path B — Azure VM (fallback / debugging)

```bash
# Local
bash research/cloud/azure/scripts/azure_provision.sh
# follow printed instructions to ssh into the VM

# On the VM
git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
cd TropFormer

# Mount blob storage (export the storage key from azure_provision output)
STORAGE_ACCOUNT=tropformerblob \
STORAGE_CONTAINER=tropformer \
STORAGE_KEY=...                 \
  bash research/cloud/azure/scripts/blob_mount.sh

# Install python deps
pip install -r research/cloud/requirements_cuda.txt
pip install datasets zstandard pyyaml huggingface_hub

# Run the full pipeline (resumable; staged in blob)
bash research/cloud/azure/scripts/run_full_pipeline.sh
```

## Stage details

### Pretraining

* **92B tokens** distributed by `weight` in `pretrain_mix.yaml`.
* Streaming dataset (`TokenShardStream`) samples each step's window from a
  random dataset proportional to weight, then a random offset inside that
  dataset's mmap'd `.bin`.
* Effective batch (8x H100, default): 8 (micro) × 4 (accum) × 8 (world) × 2048 (ctx) ≈ **524K tokens / step**.
* 92B / 524K ≈ **175K steps**; config schedules **180K** with cosine warm-up + decay.
* Maslov-h cycling on (NEXUS innovation), gist buffer reset every 100 steps.
* Best checkpoint saved as `<output_dir>/<run_tag>_best.pt`.

### SFT (curriculum)

Training proceeds **sequentially** through priority groups:

1. **reasoning** (~2.9M examples) — DeepSeek-R1, MoT, MetaMathQA, OpenMathInstruct...
2. **instruction** (~2.4M) — OpenHermes, Tulu-3, WizardLM, Magpie, SlimOrca
3. **tool** (~1.1M) — Orca-Agent, Glaive, xLAM, Hermes, Code-Feedback
4. **kyro** (~98K) — proprietary; loaded from `data/sft/kyro_*.jsonl`

Each group's step count is proportional to its example count. Within a
group the streaming dataset samples by `weight`. Per-example
`sample_weight_multiplier` (set to `5.0` for Kyro entries) scales the loss
contribution.

Prompt tokens are masked to `IGNORE` (-100) in labels; only assistant
output is supervised.

### DPO

* 25K preference pairs from `dpo_mix.yaml`. Auto-rank step (downstream of
  SFT) is configured in the manifest's `auto_rank` block — implement as a
  separate AML job that loads the SFT model and produces the filtered
  pairs.
* Policy + frozen reference both initialize from the SFT best.pt.
* DPO loss with `beta=0.1`. Logs `accuracy` (% of pairs where margin > 0)
  and `margin_mean`.

## Where to put proprietary Kyro data

Drop JSONL files at:

```
{blob_root}/raw/sft/kyro_synthetic_tools/kyro_synthetic_tools.jsonl
{blob_root}/raw/sft/kyro_temporal_reasoning/kyro_temporal_reasoning.jsonl
{blob_root}/raw/sft/kyro_memory_reasoning/...
{blob_root}/raw/sft/kyro_uncertainty/...
{blob_root}/raw/sft/kyro_consequence/...
{blob_root}/raw/sft/kyro_voice_patterns/...
{blob_root}/raw/dpo/kyro_voice_brevity/...
{blob_root}/raw/dpo/human_reviewed_ambiguity/...
```

The download manifest entries with `local_path: data/sft/...` will pick
them up; mirror that path under your blob mount.

## Cost (rough)

| Stage    | SKU                     | Time      | $/h spot | Total |
|----------|-------------------------|-----------|----------|-------|
| Download | Standard_F16s_v2 (CPU)  | ~24 h     | ~0.20    | ~$5   |
| TMT      | Standard_NC4as_T4_v3    | ~2 h      | ~0.15    | ~$0.5 |
| Tokenize | Standard_F32s_v2 (CPU)  | ~12 h     | ~0.40    | ~$5   |
| Pretrain | ND96isr_H100_v5 (8xH100)| ~5 days   | ~10      | ~$1.2K|
| SFT      | ND96isr_H100_v5         | ~36 h     | ~10      | ~$360 |
| DPO      | ND96isr_H100_v5         | ~6 h      | ~10      | ~$60  |

(Prices vary. Spot eviction can extend wall-clock; all stages resume.)
