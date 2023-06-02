#include <ssconv.hpp>
#include<ssconv.cuh>

template<type scalar_t>
__global__ void ssconv_forward_gpu_kernel(const int N,
                                          const T* input,
                                          const INDEX_TYPE* in_index,
                                          const T* weight,
                                          const T* bias,
                                          T* output,
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
    CUDA_1D_KERNEL_LOOP(index, N)
    {
        long idx_o    = index;
        const long oc = idx_o % out_channels;
        idx_o /= out_channels;
        const long ow = idx_o % out_width;
        idx_o /= out_width;
        const long oh = idx_o % out_height;
        idx_o /= out_height;
        const long bs = idx;

        T max_val = std::numeric_limits<T>::lowest(), val;
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

__global__ void ssconv_forward_gpu(Tensor input, Tensor in_index, Tensor weight, Tensor bias, Tensor output, Tensor out_index,
                                   long batch_size, long in_height, long in_width, long in_channels, long in_groups,
                                   long kernel_h, long kernel_w, long stride_h, long stride_w, 
                                   long out_height, long out_width, long out_channels, long out_groups)
{// dispatch type(float or half)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ssconv_forward_cpu", [&]
    {
        int N = batch_size * out_height * out_width * out_channels;
        ssconv_forward_gpu_kernel<scalar_t><<<GET_BLOCKS(N), THREADS_PER_BLOCK>>>(
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