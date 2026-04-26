"""TropFormer cloud-training package.

CUDA / NVIDIA-cloud port of the research-branch DirectML stack.
Primary entry-points:
    research.cloud.train_cloud      — auto-detecting trainer (single GPU / DDP)
    research.cloud.tsrn_cuda        — CUDA-fast-path helpers (drop-in for tsrn_dml)

Designed to run on free-tier credits across:
    - Modal Labs (serverless)
    - RunPod / Lambda Labs / vast.ai (rented pods)
    - Raw SSH on any CUDA box (Dockerfile or pip)
"""
