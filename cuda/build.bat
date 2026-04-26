@echo off
REM Build TropFormer CUDA kernels for Windows
REM Requires: CUDA toolkit (nvcc) in PATH, Visual Studio C++ compiler

set NVCC_OPTS=-O3 --use_fast_math -arch=sm_70 -Xcompiler /MD

REM Build max-plus GEMM kernel (existing)
nvcc -shared -o maxplus_gemm.dll maxplus_gemm.cu %NVCC_OPTS% ^
    -gencode=arch=compute_70,code=sm_70 ^
    -gencode=arch=compute_75,code=sm_75 ^
    -gencode=arch=compute_80,code=sm_80 ^
    -gencode=arch=compute_86,code=sm_86 ^
    -gencode=arch=compute_89,code=sm_89
if %ERRORLEVEL% neq 0 goto :error
echo [OK] maxplus_gemm.dll

REM Build prefix-max scan kernel (TSRN TropicalSSM accelerator)
nvcc -shared -o prefix_max_scan.dll prefix_max_scan.cu %NVCC_OPTS% ^
    -gencode=arch=compute_70,code=sm_70 ^
    -gencode=arch=compute_75,code=sm_75 ^
    -gencode=arch=compute_80,code=sm_80 ^
    -gencode=arch=compute_86,code=sm_86 ^
    -gencode=arch=compute_89,code=sm_89
if %ERRORLEVEL% neq 0 goto :error
echo [OK] prefix_max_scan.dll

REM Build tropical SSM fused kernel (state recurrence)
nvcc -shared -o tropical_ssm_fused.dll tropical_ssm_fused.cu %NVCC_OPTS% ^
    -gencode=arch=compute_70,code=sm_70 ^
    -gencode=arch=compute_75,code=sm_75 ^
    -gencode=arch=compute_80,code=sm_80 ^
    -gencode=arch=compute_86,code=sm_86 ^
    -gencode=arch=compute_89,code=sm_89
if %ERRORLEVEL% neq 0 goto :error
echo [OK] tropical_ssm_fused.dll

echo.
echo All CUDA kernels built successfully.
goto :eof

:error
echo.
echo ERROR: Build failed. Ensure nvcc and cl.exe are in PATH.
exit /b 1
