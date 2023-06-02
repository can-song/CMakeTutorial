#ifndef CORE_SSCONV_HPP
#define CORE_SSCONV_HPP

#include <torch/extension.h>
#include <vector>
using namespace at;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)

#define INDEX_TYPE short

#define DEBUG

#endif // CORE_SSCONV_HPP