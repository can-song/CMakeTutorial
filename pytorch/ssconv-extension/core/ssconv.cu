#include <ssconv.hpp>
#include<ssconv.cuh>
#include <torch/types.h>

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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ssconv_forward_cpu", [&]
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
}

template <typename scalar_t>
__global__ void ssconv_backward_input_gpu_kernel(const int N,
                                                 const scalar_t* output_grad,
                                                 const scalar_t* input,
                                                 const INDEX_TYPE* in_index,
                                                 const scalar_t* weight,
                                                 const INDEX_TYPE* out_index,
                                                 scalar_t* input_grad,
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
        long idx_i    = index;
        const long ic = idx_i % in_channels;
        idx_i /= in_channels;
        const long iw = idx_i % in_width;
        idx_i /= in_width;
        const long ih = idx_i % in_height;
        idx_i /= in_height;
        const long bs = idx_i;
        const ig = in_index[idx_i]

        for(long oc=0; oc<out_channels; ++oc)
        {
            for(long kh=0; kh<kernel_h; ++kh)
            {
                long oh = ih + kh - pad_h;
                if(oh<0 || oh>=kernel_h || oh%stride_h) continue;
                for(long kw=0; kw<kenel_w; ++kw)
                {
                    long ow = iw + kw - pad_w;
                    if(ow<0 || ow>=kernel_w || ow%stride_w) continue;

                    long idx_o = ((bs*out_height+oh)*out_width+ow)*out_channels+oc;
                    long og    = out_index[idx_o];
                    long idx_k = (((((oc*out_groups)+og)*in_channels+ic)*in_groups+ig)*kernel_h+kh)*kernel_w+kw;

                    input_grad[idx_i] += output_grad[idx_o] * weight[idx_k];
                }
            }
        }
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
        long idx_k    = index;
        const long kw = idx_k % kernel_w;
        idx_k        /= kernel_w;
        const long kh = idx_k % kernel_h;
        idx_k        /= kernel_h;
        const long ig = idx_k % in_groups;
        idx_k        /= in_groups;
        const long ic = idx_k % in_channels;
        idx_k        /= in_channels;
        const long og = idx_k % out_groups;
        idx_k        /= out_groups;
        const long oc = idx_k;

        for(long bs=0; bs<batch_size; ++bs)
        {
            long idx_i = (((bs*in_height+ih)*in_width+iw)*in_channels+ic;
            long idx_o = (((bs*out_height+oh)*out_width+ow)*out_channels+oc);
            weight_grad[idx_k] += output_grad[idx_o] * input[idx_i];
        }
    }
}

template <typename scalar_t>
__global__ void ssconv_backward_bias_gpu_kernel(const int N,
                                                const scalar_t* output_grad,
                                                const scalar_t* input,
                                                const INDEX_TYPE* in_index,
                                                const scalar_t* weight,
                                                const INDEX_TYPE* out_index,
                                                scalar_t bias_grad,
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
        long idx_b    = index;
        const long og = idx_b % out_groups;
        idx_b /= out_groups;
        const long oc = idx_b;

        for(long bs=0; bs<batch_size; ++bs)
        {
            for(long oh=0; oh<out_height; ++oh)
            {
                for(long ow=0; ow<out_width; ++ow)
                {
                    long idx_o = ((bs*out_height+oh)*out_width+ow)*out_channels+oc;
                    if(og!=out_index[idx_o]) continue;
                    bias_grad[idx_b] += out_grad[idx_o];
                }
            }
        }
    }
}


std::vector<Tensor>  ssconv_backward_gpu(Tensor output_grad, Tensor input, Tensor in_index, 
                                         Tensor weight, Tensor bias, Tensor out_index,
                                         long batch_size, long in_height, long in_width, long in_channels, long in_groups,
                                         long kernel_h, long kernel_w, long stride_h, long stride_w, 
                                         long out_height, long out_width, long out_channels, long out_groups)
{
    Tensor input_grad  = at::zeros_like(input);
    Tensor weight_grad = at::zeros_like(weight);
    Tensor bias_grad   = at::zeros_like(bias);

    // dispatch type(float or half)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ssconv_forward_cpu", [&]
    {
        int N = input_grad.numel();
        ssconv_backward_input_gpu_kernel<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, output_grad, input, in_index, weight, out_index, input_grad,
            batch_size, in_height, in_width, in_channels, in_groups, 
            kernel_h, kernel_w, stride_h, stride_w,
            out_height, out_width, out_channels, out_groups
        );
        N = weight_grad.numel();
        ssconv_backward_weight_gpu_kernel<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, output_grad, input, in_index, weight, out_index, weight_grad,
            batch_size, in_height, in_width, in_channels, in_groups, 
            kernel_h, kernel_w, stride_h, stride_w,
            out_height, out_width, out_channels, out_groups
        );
        N = bias_grad.numel();
        ssconv_backward_bias_gpu_kernel<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            N, output_grad, input, in_index, weight, out_index, bias_grad,
            batch_size, in_height, in_width, in_channels, in_groups, 
            kernel_h, kernel_w, stride_h, stride_w,
            out_height, out_width, out_channels, out_groups
        );
    });

    return {input_grad, weight_grad, bias_grad}
}