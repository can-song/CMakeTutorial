// Modify from https://github.com/open-mmlab/mmcv/blob/my_conv/mmcv/ops/csrc/common/cuda/deform_conv_cuda_kernel.cuh
// Copyright (c) OpenMMLab. All rights reserved.
#include <torch/types.h>
#include "pytorch_cuda_helper.hpp"
template <typename T>
__global__ void my_conv_im2col_gpu_kernel(
    const int n, const T *data_im, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int batch_size,
    const int num_channels, const int height_col,
    const int width_col, T *data_col)
{
    CUDA_1D_KERNEL_LOOP(index, n)
    {
        // index index of output matrix
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int b_col = (index / width_col / height_col) % batch_size;
        const int c_im = (index / width_col / height_col) / batch_size;
        const int c_col = c_im * kernel_h * kernel_w;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        T *data_col_ptr =
            data_col +
            ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
        const T *data_im_ptr =
            data_im + (b_col * num_channels + c_im) * height * width;

        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                T val = static_cast<T>(0);
                const int h_im = h_in + i * dilation_h;
                const int w_im = w_in + j * dilation_w;
                if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
                {
                    val = data_im_ptr[h_im * width + w_im];
                }
                *data_col_ptr = val;
                data_col_ptr += batch_size * height_col * width_col;
            }
        }
    }
}

void my_conv_im2col_cuda(Tensor data_im,
                         const int channels, const int height,
                         const int width, const int ksize_h,
                         const int ksize_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int dilation_h, const int dilation_w,
                         const int parallel_imgs, Tensor data_col)
{
    int height_col =
        (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col * parallel_imgs;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_im.scalar_type(), "my_conv_im2col_gpu", [&]
        { my_conv_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels),
                                                THREADS_PER_BLOCK, 0,
                                                at::cuda::getCurrentCUDAStream()>>>(
              num_kernels, data_im.data_ptr<scalar_t>(),
              height, width, ksize_h, ksize_w,
              pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
              parallel_imgs, channels,
              height_col, width_col, data_col.data_ptr<scalar_t>()); });

    AT_CUDA_CHECK(cudaGetLastError());
}


template<typename scalar_t>
__global__ void ssconv_forward_gpu_kernel(const int N,
                                          const scalar_t* input,
                                          const INDEX_TYPE* in_index,
                                          const scalar_t* weight,
                                          const scalar_t* bias,
                                          scalar_t* output,
                                          INDEX_TYPE* out_index,
                                          long batch_size, 
                                          long in_height, 
                                          long in_width,
                                          long in_channels, 
                                          long in_groups,
                                          long kernel_h,
                                          long kernel_w,
                                          long stride_h,
                                          long stride_w,
                                          long out_height,
                                          long out_width,
                                          long out_channels,
                                          long out_groups)
{
    long pad_h = kernel_h >> 1;
    long pad_w = kernel_w >> 1;
    CUDA_1D_KERNEL_LOOP(index, N)
    {
        long idx_o    = index;
        const long oc = idx_o % out_channels;
        idx_o /= out_channels;
        const long ow = idx_o % out_width;
        idx_o /= out_width;
        const long oh = idx_o % out_height;
        idx_o /= out_height;
        const long bs = idx_o;

        scalar_t max_val = std::numeric_limits<scalar_t>::lowest(), val;
        INDEX_TYPE max_idx = 0, idx;
        for(long og = 0; og < out_groups; og++)
        {
            val = bias[oc*out_groups+og];
            idx = og;
            for(long ic=0; ic<in_channels; ic++)
            {
                for(long kh=0; kh<kernel_h; kh++)
                {
                    long ih = oh*stride_h + (kh - pad_h);
                    if (ih<0 || ih>=in_height) continue;
                    for(long kw=0; kw<kernel_w; kw++){
                        long iw = ow*stride_w + (kw - pad_w);
                        if (iw<0 || iw>=in_width) continue;

                        long idx_i = ((bs*in_height+ih)*in_width+iw)*in_channels+ic;
                        long ig = in_index[idx_i];
                        long idx_k = (((((oc*out_groups)+og)*in_channels+ic)*in_groups+ig)*kernel_h+kh)*kernel_w+kw;
                        val += input[idx_i] * weight[idx_k];
                    }
                }
            }
            if (val > max_val)
            {
                max_val = val;
                max_idx = idx;
            }
        }
        output[idx_o] = max_val;
        out_index[idx_o] = max_idx;
    }
}

void ssconv_forward_gpu(Tensor input, Tensor in_index, Tensor weight, Tensor bias, Tensor output, Tensor out_index,
    long batch_size, long in_height, long in_width, long in_channels, long in_groups,
    long kernel_h, long kernel_w, long stride_h, long stride_w, 
    long out_height, long out_width, long out_channels, long out_groups)
{
    // dispatch type(float or half)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ssconv_forward_gpu", [&]
    {
        int N = batch_size * out_height * out_width * out_channels;
        ssconv_forward_gpu_kernel<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            N,
            input.data_ptr<scalar_t>(), 
            in_index.data_ptr<INDEX_TYPE>(), 
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            out_index.data_ptr<INDEX_TYPE>(),
            batch_size,
            in_height,
            in_width,
            in_channels,
            in_groups,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            out_height,
            out_width,
            out_channels,
            out_groups);
    });

    AT_CUDA_CHECK(cudaGetLastError());
}