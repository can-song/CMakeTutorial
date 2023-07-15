#include <torch/types.h>
// #include <vector>
// #include <limits>
#include "ssconv_cuda.cuh"


template<typename scalar_t>
__global__ void ssconv_forward_gpu_kernel(const int N,
                                          const scalar_t* input,
                                          const INDEX_TYPE* in_index,
                                          const scalar_t* weight,
                                          const scalar_t* bias,
                                          scalar_t* output,
                                          INDEX_TYPE* out_index,
                                          const long batch_size, 
                                          const long in_height, 
                                          const long in_width, 
                                          const long in_channels, 
                                          const long in_block_size,
                                          const long kernel_h,
                                          const long kernel_w,
                                          const long stride_h,
                                          const long stride_w,
                                          const long pad_h,
                                          const long pad_w,
                                          const long out_height,
                                          const long out_width,
                                          const long out_channels,
                                          const long out_block_size)
{
    const long in_groups = in_block_size;
    const long out_groups = out_block_size;
    CUDA_1D_KERNEL_LOOP(index, N)
    {
        const long idx_o    = index;
        const long ow = index % out_width;
        index /= out_width;
        const long oh = index % out_height;
        index /= out_height;
        const long oc = index % out_channels;
        index /= out_channels;
        const long bs = index;

        scalar_t max_val = std::numeric_limits<scalar_t>::lowest(), val;
        INDEX_TYPE max_idx = 0, idx;
        for(INDEX_TYPE og = 0; og < out_groups; ++og)
        {
            val = bias[oc*out_groups+og];
            idx = og;
            for(long ic=0; ic<in_channels; ++ic)
            {
                for(long kh=0; kh<kernel_h; ++kh)
                {
                    long ih = oh*stride_h + (kh - pad_h);
                    if (ih<0 || ih>=in_height) continue;
                    for(long kw=0; kw<kernel_w; ++kw){
                        long iw = ow*stride_w + (kw - pad_w);
                        if (iw<0 || iw>=in_width) continue;

                        const long idx_i = ((bs*in_channels+ic)*in_height+ih)*in_width+iw;
                        INDEX_TYPE ig = in_index[idx_i];
                        long idx_k = ((((oc*out_groups+og)*in_channels+ic)*in_groups+ig)*kernel_h+kh)*kernel_w+kw;
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
    long kernel_h, long kernel_w, long stride_h, long stride_w, long pad_h, long pad_w,
    long out_height, long out_width, long out_channels, long out_groups)
{
#ifndef NDEBUG
    fprintf(stdout, "out_groups: %d\n", out_groups);
#endif
    // dispatch type(float or half)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ssconv_forward_gpu", [&]
    {
        // int N = batch_size * out_height * out_width * out_channels;
        int N = output.numel();
#ifndef NDEBUG
        fprintf(stdout, "N: %d\n", N);
#endif
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
            pad_h,
            pad_w,
            out_height,
            out_width,
            out_channels,
            out_groups);
    });

    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
__global__ void ssconv_backward_input_gpu_kernel(const int N,
                                                 const scalar_t* output_grad,
                                                 const scalar_t* input,
                                                 const INDEX_TYPE* in_index,
                                                 const scalar_t* weight,
                                                 const INDEX_TYPE* out_index,
                                                 scalar_t* input_grad,
                                                 const long batch_size, 
                                                 const long in_height, 
                                                 const long in_width,
                                                 const long in_channels, 
                                                 const long in_block_size,
                                                 const long kernel_h,
                                                 const long kernel_w,
                                                 const long stride_h,
                                                 const long stride_w,
                                                 const long pad_h,
                                                 const long pad_w,
                                                 const long out_height,
                                                 const long out_width,
                                                 const long out_channels,
                                                 const long out_block_size)
{
    const long in_groups = in_block_size;
    const long out_groups = out_block_size;
    CUDA_1D_KERNEL_LOOP(index, N)
    {
        const long idx_i    = index;
        const long iw = index % in_width;
        index /= in_width;
        const long ih = index % in_height;
        index /= in_height;
        const long ic = index % in_channels;
        index /= in_channels;
        const long bs = index;
        const long ig = in_index[idx_i];

        scalar_t grad = 0;
        for(long oc=0; oc<out_channels; ++oc)
        {
            for(long kh=0; kh<kernel_h; ++kh)
            {
                long oh = ih - (kh - pad_h);
                if(oh%stride_h) continue;
                oh /= stride_h;
                if((oh<0) || (oh>=out_height)) continue;
                for(long kw=0; kw<kernel_w; ++kw)
                {
                    long ow = iw - (kw - pad_w);
                    if(ow%stride_w) continue;
                    ow /= stride_w;
                    if((ow<0) || (ow>=out_width)) continue;
                    const long idx_o = ((bs*out_channels+oc)*out_height+oh)*out_width+ow;
                    long og    = out_index[idx_o];
                    long idx_k = (((((oc*out_groups)+og)*in_channels+ic)*in_groups+ig)*kernel_h+kh)*kernel_w+kw;

                    grad += output_grad[idx_o] * weight[idx_k];
                }
            }
        }
        input_grad[idx_i] += grad;
    }
}

template <typename scalar_t>
__global__ void ssconv_backward_weight_gpu_kernel(const int N,
                                                  const scalar_t* output_grad,
                                                  const scalar_t* input,
                                                  const INDEX_TYPE* in_index,
                                                  const scalar_t* weight,
                                                  const INDEX_TYPE* out_index,
                                                  scalar_t* weight_grad,
                                                  const long batch_size, 
                                                  const long in_height, 
                                                  const long in_width,
                                                  const long in_channels, 
                                                  const long in_block_size,
                                                  const long kernel_h,
                                                  const long kernel_w,
                                                  const long stride_h,
                                                  const long stride_w,
                                                  const long pad_h,
                                                  const long pad_w,
                                                  const long out_height,
                                                  const long out_width,
                                                  const long out_channels,
                                                  const long out_block_size)
{
    const long in_groups = in_block_size;
    const long out_groups = out_block_size;

    CUDA_1D_KERNEL_LOOP(index, N)
    {
        const long idx_k    = index;
        const long kw = index % kernel_w;
        index        /= kernel_w;
        const long kh = index % kernel_h;
        index        /= kernel_h;
        const long ig = index % in_groups;
        index        /= in_groups;
        const long ic = index % in_channels;
        index        /= in_channels;
        const long og = index % out_groups;
        index        /= out_groups;
        const long oc = index;

        scalar_t grad = 0;
        for(long bs=0; bs<batch_size; ++bs)
        {
            for(long oh=0; oh<out_height; ++oh)
            {
                long ih = oh*stride_h + (kh - pad_h);
                if(ih<0 || ih>=in_height) continue;
                for(long ow=0; ow<out_width; ++ow)
                {
                    long iw = ow*stride_w + (kw - pad_w);
                    if(iw<0 || iw>=in_width) continue;
                    const long idx_i = ((bs*in_channels+ic)*in_height+ih)*in_width+iw;
                    const long idx_o = ((bs*out_channels+oc)*out_height+oh)*out_width+ow;
                    if((in_index[idx_i]==ig) && (out_index[idx_o]==og))
                        grad += output_grad[idx_o] * input[idx_i];
                }
            }
        }
        weight_grad[idx_k] += grad;
    }
}

template <typename scalar_t>
__global__ void ssconv_backward_bias_gpu_kernel(const int N,
                                                const scalar_t* output_grad,
                                                const scalar_t* input,
                                                const INDEX_TYPE* in_index,
                                                const scalar_t* weight,
                                                const INDEX_TYPE* out_index,
                                                scalar_t* bias_grad,
                                                const long batch_size, 
                                                const long in_height, 
                                                const long in_width,
                                                const long in_channels, 
                                                const long in_block_size,
                                                const long kernel_h,
                                                const long kernel_w,
                                                const long stride_h,
                                                const long stride_w,
                                                const long pad_h,
                                                const long pad_w,
                                                const long out_height,
                                                const long out_width,
                                                const long out_channels,
                                                const long out_block_size)
{
    const long in_groups = in_block_size;
    const long out_groups = out_block_size;

    CUDA_1D_KERNEL_LOOP(index, N)
    {
        const long idx_b    = index;
        const long og = index % out_groups;
        index /= out_groups;
        const long oc = index;

        scalar_t grad = 0;
        for(long bs=0; bs<batch_size; ++bs)
        {
            for(long oh=0; oh<out_height; ++oh)
            {
                for(long ow=0; ow<out_width; ++ow)
                {
                    const long idx_o = ((bs*out_channels+oc)*out_height+oh)*out_width+ow;
                    if(og==out_index[idx_o])
                        grad += output_grad[idx_o];
                }
            }
        }
        bias_grad[idx_b] += grad;
    }
}


std::vector<Tensor>  ssconv_backward_gpu(Tensor output_grad, Tensor input, Tensor in_index, 
                                         Tensor weight, Tensor bias, Tensor out_index,
                                         long batch_size, long in_height, long in_width, long in_channels, long in_block_size,
                                         long kernel_h, long kernel_w, long stride_h, long stride_w, long pad_h, long pad_w,
                                         long out_height, long out_width, long out_channels, long out_block_size)
{
    Tensor input_grad  = at::zeros_like(input);
    Tensor weight_grad = at::zeros_like(weight);
    Tensor bias_grad   = at::zeros_like(bias);

    // dispatch type(float or half)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ssconv_backward_gpu", [&]
    {
        int N = input_grad.numel();
        ssconv_backward_input_gpu_kernel<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, output_grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), in_index.data_ptr<INDEX_TYPE>(), 
            weight.data_ptr<scalar_t>(), out_index.data_ptr<INDEX_TYPE>(), input_grad.data_ptr<scalar_t>(),
            batch_size, in_height, in_width, in_channels, in_block_size, 
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            out_height, out_width, out_channels, out_block_size
        );
        N = weight_grad.numel();
        ssconv_backward_weight_gpu_kernel<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, output_grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), in_index.data_ptr<INDEX_TYPE>(), 
            weight.data_ptr<scalar_t>(), out_index.data_ptr<INDEX_TYPE>(), weight_grad.data_ptr<scalar_t>(),
            batch_size, in_height, in_width, in_channels, in_block_size, 
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            out_height, out_width, out_channels, out_block_size
        );
        N = bias_grad.numel();
        ssconv_backward_bias_gpu_kernel<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, output_grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), in_index.data_ptr<INDEX_TYPE>(), 
            weight.data_ptr<scalar_t>(), out_index.data_ptr<INDEX_TYPE>(), bias_grad.data_ptr<scalar_t>(),
            batch_size, in_height, in_width, in_channels, in_block_size, 
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            out_height, out_width, out_channels, out_block_size
        );
    });

    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());
    return {input_grad, weight_grad, bias_grad};
}
