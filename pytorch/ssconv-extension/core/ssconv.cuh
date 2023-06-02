#ifndef CORE_SSCONV_CUH
#define CORE_SSCONV_CUH

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <cuda.h>
using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)

// #define CUDA_1D_KERNEL_LOOP(i, n)                              \
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
//        i += blockDim.x * gridDim.x)

// #define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
//   for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
//        i += blockDim.x * gridDim.x)                                 \
//     for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
//          j += blockDim.y * gridDim.y)

// #define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)          \
//   for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
//     for (size_t j = blockIdx.y; j < (m); j += gridDim.y)

#define THREADS_PER_BLOCK 512

#endif // CORE_SSCONV_CUH