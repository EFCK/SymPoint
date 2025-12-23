#!/bin/bash
# Environment setup for SymPoint project

conda activate symp
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export NVCC_APPEND_FLAGS="-ccbin $CXX"
