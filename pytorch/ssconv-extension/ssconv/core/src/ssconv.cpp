#include "ssconv.hpp"
// #include <limits>
// #include <iostream>

void ssconv_forward_gpu(Tensor input, Tensor in_index, Tensor weight, Tensor bias, Tensor output, Tensor out_index,
                                   long batch_size, long in_height, long in_width, long in_channels, long in_block_size,
                                   long kernel_h, long kernel_w, long stride_h, long stride_w, long pad_h, long pad_w,
                                   long out_height, long out_width, long out_channels, long out_block_size);

std::vector<Tensor>  ssconv_backward_gpu(Tensor output_grad, Tensor input, Tensor in_index, 
                                         Tensor weight, Tensor bias, Tensor out_index,
                                         long batch_size, long in_height, long in_width, long in_channels, long in_block_size,
                                         long kernel_h, long kernel_w, long stride_h, long stride_w, long pad_h, long pad_w,
                                         long out_height, long out_width, long out_channels, long out_block_size);


template <typename T>
void ssconv_forward_cpu_kernel(const T* input,
                               const INDEX_TYPE* in_index,
                               const T* weight,
                               const T* bias,
                               T* output,
                               INDEX_TYPE* out_index,
                               long batch_size, 
                               long in_height, 
                               long in_width,
                               long in_channels, 
                               long in_block_size,
                               long kernel_h,
                               long kernel_w,
                               long stride_h,
                               long stride_w,
                               long pad_h,
                               long pad_w,
                               long out_height,
                               long out_width,
                               long out_channels,
                               long out_block_size)
{
    long in_groups = in_block_size;
    long out_groups = out_block_size;

    for(long bs = 0; bs < batch_size; bs++)
    {
        for(long oc = 0; oc < out_channels; oc++)
        {
            for(long oh = 0; oh < out_height; oh++)
            {
                for(long ow = 0; ow < out_width; ow++)
                {
                    const long idx_o = ((bs*out_channels+oc)*out_height+oh)*out_width+ow;
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

                                    const long idx_i = ((bs*in_channels+ic)*in_height+ih)*in_width+iw;
                                    const long ig = in_index[idx_i];
                                    const long idx_k = (((((oc*out_groups)+og)*in_channels+ic)*in_groups+ig)*kernel_h+kh)*kernel_w+kw;
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
        }

    }
}

void ssconv_forward_cpu(Tensor input, Tensor in_index, Tensor weight, Tensor bias, Tensor output, Tensor out_index,
                        long batch_size, long in_height, long in_width, long in_channels, long in_block_size,
                        long kernel_h, long kernel_w, long stride_h, long stride_w, long pad_h, long pad_w,
                        long out_height, long out_width, long out_channels, long out_block_size)
{
    // dispatch type(float or half)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ssconv_forward_cpu", [&]
    {
        ssconv_forward_cpu_kernel<scalar_t>(input.data_ptr<scalar_t>(), 
                                            in_index.data_ptr<INDEX_TYPE>(), 
                                            weight.data_ptr<scalar_t>(),
                                            bias.data_ptr<scalar_t>(),
                                            output.data_ptr<scalar_t>(),
                                            out_index.data_ptr<INDEX_TYPE>(),
                                            batch_size,
                                            in_height,
                                            in_width,
                                            in_channels,
                                            in_block_size,
                                            kernel_h,
                                            kernel_w,
                                            stride_h,
                                            stride_w,
                                            pad_h,
                                            pad_w,
                                            out_height,
                                            out_width,
                                            out_channels,
                                            out_block_size);
    });
}

void ssconv_forward(Tensor input,
                    Tensor in_index,
                    Tensor weight,
                    Tensor bias,
                    Tensor output,
                    Tensor out_index,
                    long in_block_size,
                    long out_block_size,
                    long stride_h,
                    long stride_w,
                    long pad_h,
                    long pad_w)
{
    bool is_cuda = false;
    if(input.device().is_cuda())
    {
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(in_index);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(bias);
        CHECK_CUDA_INPUT(output);
        is_cuda = true;
    }
    else
    {
        CHECK_CPU_INPUT(input);
        CHECK_CPU_INPUT(in_index);
        CHECK_CPU_INPUT(weight);
        CHECK_CPU_INPUT(bias);
        CHECK_CPU_INPUT(output);
        is_cuda = false;
    }

    //TODO: check shape

    long batch_size   = input.size(0);
    long in_height    = input.size(2);
    long in_width     = input.size(3);

    long out_channels = weight.size(0) / out_block_size;
    long in_channels  = weight.size(1) / in_block_size;
    long kernel_h     = weight.size(2);
    long kernel_w     = weight.size(3);

    long out_height   = output.size(2);
    long out_width    = output.size(3);

#ifndef NDEBUG
    std::cout << "out_block_size: " << out_block_size << std::endl
              << "in_block_size: " << in_block_size << std::endl;
#endif // !NDEBUG



    // dispatch device(cuda or cpu)
    if(is_cuda)
    {
        ssconv_forward_gpu(input, in_index,  weight, bias, output, out_index,
                           batch_size, in_height, in_width, in_channels, in_block_size, 
                           kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                           out_height, out_width, out_channels, out_block_size);
    }
    else
    {
        ssconv_forward_cpu(input, in_index,  weight, bias, output, out_index,
                           batch_size, in_height, in_width, in_channels, in_block_size, 
                           kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                           out_height, out_width, out_channels, out_block_size);
    }
}

template <typename T>
void ssconv_backward_cpu_kernel(const T* output_grad,
                                const T* input,
                                const INDEX_TYPE* in_index,
                                const T* weight,
                                const INDEX_TYPE* out_index,
                                T* input_grad,
                                T* weight_grad,
                                T* bias_grad,
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
#ifndef NDEBUG
    std::cout << "pad_h:       " << pad_h << std::endl
              << "pad_w:       " << pad_w << std::endl
              << "kernel_h:    " << kernel_h << std::endl
              << "kernel_w:    " << kernel_w << std::endl
              << "stride_h:    " << stride_h << std::endl
              << "stride_w:    " << stride_w << std::endl
              << "in_height:   " << in_height << std::endl
              << "in_width:    " << in_width << std::endl
              << "in_channels: " << in_channels << std::endl
              << "in_groups:   " << in_groups << std::endl
              << "out_height:  " << out_height << std::endl
              << "out_width:   " << out_width << std::endl
              << "out_channels:" << out_channels << std::endl;
#endif // NDEBUG

for(long bs = 0; bs < batch_size; bs++)
    {
        for(long oh = 0; oh < out_height; oh++)
        {
            for(long ow = 0; ow < out_width; ow++)
            {
                for(long oc = 0; oc < out_channels; oc++)
                {
                    // long idx_o                   = ((bs * out_height + oh)*out_width + ow)*out_channels + oc;
                    const long idx_o             = ((bs*out_channels+oc)*out_height+oh)*out_width + ow;
                    long og                      = out_index[idx_o];
                    bias_grad[oc*out_groups+og] += output_grad[idx_o];
                    for(long ic=0; ic<in_channels; ic++)
                    {
                        for(long kh=0; kh<kernel_h; kh++)
                        {
                            long ih = oh*stride_h + (kh - pad_h);
                            if (ih<0 || ih>=in_height) continue;
                            for(long kw=0; kw<kernel_w; kw++)
                            {
                                long iw = ow*stride_w + (kw - pad_w);
                                if (iw<0 || iw>=in_width) continue;

                                // long idx_i          = ((bs*in_height+ih)*in_width+iw)*in_channels+ic;
                                const long idx_i    = ((bs*in_channels+ic)*in_height+ih)*in_width + iw;
                                long ig             = in_index[idx_i];
                                long idx_k          = (((((oc*out_groups)+og)*in_channels+ic)*in_groups+ig)*kernel_h+kh)*kernel_w+kw;

                                weight_grad[idx_k] += output_grad[idx_o] * input[idx_i];
                                input_grad[idx_i]  += output_grad[idx_o] * weight[idx_k];
                            }
                        }
                    }     
                }
            }
        }

    }

}

std::vector<Tensor>  ssconv_backward_cpu(Tensor output_grad, Tensor input, Tensor in_index, 
                                         Tensor weight, Tensor bias, Tensor out_index,
                                         long batch_size, long in_height, long in_width, long in_channels, long in_block_size,
                                         long kernel_h, long kernel_w, long stride_h, long stride_w, long pad_h, long pad_w,
                                         long out_height, long out_width, long out_channels, long out_block_size)
{
    Tensor input_grad  = at::zeros_like(input);
    Tensor weight_grad = at::zeros_like(weight);
    Tensor bias_grad   = at::zeros_like(bias);

    // dispatch type(float or half)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ssconv_forward_cpu", [&]
    {
        ssconv_backward_cpu_kernel<scalar_t>(output_grad.data_ptr<scalar_t>(),
                                             input.data_ptr<scalar_t>(), 
                                             in_index.data_ptr<INDEX_TYPE>(), 
                                             weight.data_ptr<scalar_t>(),
                                             out_index.data_ptr<INDEX_TYPE>(),
                                             input_grad.data_ptr<scalar_t>(),
                                             weight_grad.data_ptr<scalar_t>(),
                                             bias_grad.data_ptr<scalar_t>(),
                                             batch_size,
                                             in_height,
                                             in_width,
                                             in_channels,
                                             in_block_size,
                                             kernel_h,
                                             kernel_w,
                                             stride_h,
                                             stride_w,
                                             pad_h,
                                             pad_w,
                                             out_height,
                                             out_width,
                                             out_channels,
                                             out_block_size);
    });
    return {input_grad, weight_grad, bias_grad};
}

std::vector<Tensor> ssconv_backward(Tensor output_grad, 
                                    Tensor input,
                                    Tensor in_index,
                                    Tensor weight,
                                    Tensor bias,
                                    Tensor out_index, 
                                    const long in_block_size,
                                    const long out_block_size,
                                    const long stride_h,
                                    const long stride_w,
                                    const long pad_h,
                                    const long pad_w)
{
    // check device
    bool is_cuda = false;
    if(input.device().is_cuda())
    {
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(in_index);
        CHECK_CUDA_INPUT(weight);
        CHECK_CUDA_INPUT(bias);
        CHECK_CUDA_INPUT(output_grad);
        CHECK_CUDA_INPUT(out_index);
        is_cuda = true;
    }
    else
    {
        CHECK_CPU_INPUT(input);
        CHECK_CPU_INPUT(in_index);
        CHECK_CPU_INPUT(weight);
        CHECK_CPU_INPUT(bias);
        CHECK_CPU_INPUT(output_grad);
        CHECK_CPU_INPUT(out_index);
        is_cuda = false;
    }
    // check shape

    long batch_size   = input.size(0);
    long in_height    = input.size(2);
    long in_width     = input.size(3);

    long out_channels = weight.size(0) / out_block_size;
    long in_channels  = weight.size(1) / in_block_size;
    long kernel_h     = weight.size(2);
    long kernel_w     = weight.size(3);

    long out_height   = output_grad.size(2);
    long out_width    = output_grad.size(3);

    // dispatch device(cuda or cpu)
    if(is_cuda)
    {
        return ssconv_backward_gpu(output_grad, input, in_index, weight, bias, out_index,
                                   batch_size, in_height, in_width, in_channels, in_block_size, 
                                   kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                                   out_height, out_width, out_channels, out_block_size);
    }
    else
    {
        return ssconv_backward_cpu(output_grad, input, in_index, weight, bias, out_index,
                                   batch_size, in_height, in_width, in_channels, in_block_size, 
                                   kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                                   out_height, out_width, out_channels, out_block_size);
    }

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("forward", &ssconv_forward, "sparse to sparse convolution forward",
          py::arg("input"), py::arg("in_index"), 
          py::arg("weight"), py::arg("bias"), py::arg("output"), py::arg("out_index"),
          py::arg("in_block_size"), py::arg("out_block_size"), py::arg("stride_h"), py::arg("stride_w"), 
          py::arg("pad_h"), py::arg("pad_w"));
    m.def("backward", &ssconv_backward, "sparse to sparse convolution backward",
          py::arg("output_grad"), py::arg("input"), py::arg("in_index"), 
          py::arg("weight"), py::arg("bias"), py::arg("out_index"),
          py::arg("in_block_size"), py::arg("out_block_size"), py::arg("stride_h"), py::arg("stride_w"),
          py::arg("pad_h"), py::arg("pad_w"));
}