#!/bin/bash
# Build the max-plus GEMM CUDA kernel
# Requires: CUDA toolkit (nvcc)

# Linux
nvcc -shared -o maxplus_gemm.so maxplus_gemm.cu \
    -Xcompiler -fPIC \
    -O3 \
    --use_fast_math \
    -arch=sm_70 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86

echo "Built maxplus_gemm.so"

# To build on Windows:
# nvcc -shared -o maxplus_gemm.dll maxplus_gemm.cu -O3 --use_fast_math -arch=sm_70
