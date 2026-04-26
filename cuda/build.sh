#!/bin/bash
# Build TropFormer CUDA kernels
# Requires: CUDA toolkit (nvcc)

COMMON="-Xcompiler -fPIC -O3 --use_fast_math -arch=sm_70"
GENCODES="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89"

# 1. Max-plus GEMM (existing)
nvcc -shared -o maxplus_gemm.so maxplus_gemm.cu $COMMON $GENCODES
echo "[OK] maxplus_gemm.so"

# 2. Prefix-max scan (TropicalSSM / prefix_max replacement)
nvcc -shared -o prefix_max_scan.so prefix_max_scan.cu $COMMON $GENCODES
echo "[OK] prefix_max_scan.so"

# 3. Tropical SSM fused recurrence
nvcc -shared -o tropical_ssm_fused.so tropical_ssm_fused.cu $COMMON $GENCODES
echo "[OK] tropical_ssm_fused.so"

echo "All CUDA kernels built."

# Windows: run build.bat instead
