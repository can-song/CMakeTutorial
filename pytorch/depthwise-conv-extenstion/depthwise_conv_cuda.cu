template <typename scalar_t>
__global__ void ConvForward(const int nthreads,
		const scalar_t* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		scalar_t* const top_data,const scalar_t* const weight,const scalar_t* const bias,const bool bias_term_) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % conved_width;
		const int ph = (index / conved_width) % conved_height;
		const int c = (index / conved_width / conved_height) % channels;
		const int n = index / conved_width / conved_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
		scalar_t aveval = 0;
		const scalar_t* const bottom_slice =
		bottom_data + (n * channels + c) * height * width;
		const scalar_t* const weight_slice =
		weight + c * kernel_h * kernel_w;
		int khstart=hend<kernel_h?kernel_h-hend:0;
		int kwstart=wend<kernel_w?kernel_w-wend:0;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {

				aveval += bottom_slice[h * width + w]*weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];

			}
		}
		if(bias_term_) {
			aveval+=bias[c];
		}
		top_data[index] = aveval;
	}
}


//对输入张量（Tensor）、权重Weight和Bias实现backward过程。
template <typename scalar_t>
__global__ void ConvBackward(const int nthreads,
const scalar_t* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
scalar_t* const bottom_diff,
const scalar_t* const weight) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w = index % width + pad_w;
		const int h = (index / width) % height + pad_h;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;

		const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
		const int phend = min(h / stride_h + 1, conved_height);
		const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
		const int pwend = min(w / stride_w + 1, conved_width);

		const int khstart=(h >= kernel_h) ? ((h-kernel_h)%stride_h)+(kernel_h-stride_h): h;
		const int kwstart=(w >= kernel_w) ? ((w-kernel_w)%stride_w)+(kernel_w-stride_w) : w;

		scalar_t gradient = 0;
		const scalar_t* const top_diff_slice =
		top_diff + (n * channels + c) * conved_height * conved_width;

		const scalar_t* const weight_slice =weight + c * kernel_h * kernel_w;

		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				int kh=khstart-(ph-phstart)*stride_h;
				int kw=kwstart-(pw-pwstart)*stride_w;
				gradient += top_diff_slice[ph * conved_width + pw] *weight_slice[kh*kernel_w+kw];
			}
		}
		bottom_diff[index] = gradient;
	}
}



template <typename scalar_t>
__global__ void ConvBackwardWeight(const int nthreads,
const scalar_t* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
scalar_t* const weight_diff,
const scalar_t* const bottom_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int kw=index % kernel_w;
		const int kh= (index /kernel_w)%kernel_h;
		const int c=index /kernel_w/kernel_h;
        scalar_t gradient = 0;
		for( int n=0;n<num;n++) {

			const scalar_t* const top_diff_slice = top_diff + (n * channels + c) * conved_height * conved_width;
			const scalar_t* const bottom_data_slice = bottom_data + (n * channels + c) * height * width;


			const int phstart=max(DIVIDE_CEIL((pad_h-kh),stride_h),0);
			const int phend=min(DIVIDE_CEIL((height+pad_h-kh),stride_h),conved_height);

			const int pwstart=max(DIVIDE_CEIL((pad_w-kw),stride_w),0);

			const int pwend=min(DIVIDE_CEIL((width+pad_w-kw),stride_w),conved_width);

			for(int ph=phstart;ph<phend;ph++){
				for (int pw=pwstart;pw<pwend;pw++){
					const int h=ph*stride_h+kh-pad_h;
					const int w=pw*stride_w+kw-pad_w;
					gradient+=top_diff_slice[ph * conved_width + pw]*bottom_data_slice[h*width+w];
				}
			}
		}
		weight_diff[c * kernel_h * kernel_w+kh*kernel_w+kw]+=gradient;
	}
}

template <typename scalar_t>
__global__ void ConvBackwardBias(const int nthreads,
const scalar_t* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
scalar_t* const bias_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int c = index;
		scalar_t gradient=0;
		for( int n=0;n<num;n++) {
			const scalar_t* const top_diff_slice =
			top_diff + (n * channels + c) * conved_height * conved_width;
			for(int ph=0;ph<conved_height;ph++) {
				for (int pw=0;pw<conved_width;pw++) {
					gradient+=top_diff_slice[ph * conved_width + pw];
				}
			}
		}
		bias_diff[c]+=gradient;
	}
}

//封装DepthWiseConvBackwarddLaucher函数调用cuda kernel 函数。
std::vector<at::Tensor> DepthWiseConvBackwarddLaucher(const at::Tensor output_grad, const at::Tensor input, const at::Tensor weight, const at::Tensor bias,
                                                      const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
                                                      const int pad_h, const int pad_w, const bool bias_term_){
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto kernal_extent_h = /* dilation_h * */ (kernel_h - 1) + 1;
    const auto conved_height = (input_height + 2 * pad_h - kernal_extent_h) / stride_h + 1;
    const auto kernal_extent_w = /* dilation_w * */ (kernel_w - 1) + 1;
    const auto conved_width = (input_width + 2 * pad_w - kernal_extent_w) / stride_w + 1;

    const int count_weight = channels * kernel_h * kernel_w;
    const int count_input = batch_size * channels * input_height * input_width;

    auto weight_diff = at::zeros_like(weight);
    auto bottom_diff = at::zeros_like(input);
    at::Tensor bias_diff;
    int count_bias = 0;

    if (bias_term_){
        count_bias = channels;
        bias_diff = at::zeros_like(bias);
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        output_grad.type(), "ConvLaucherBackward",
        ([&]{
            const scalar_t *bottom_data = input.data<scalar_t>();
            const scalar_t *depthwise_weight = weight.data<scalar_t>();
            const scalar_t *top_diff = output_grad.data<scalar_t>();
            scalar_t *depthwise_weight_diff = weight_diff.data<scalar_t>();
            scalar_t *depthwise_bottom_diff = bottom_diff.data<scalar_t>();

            if (bias_term_){
                scalar_t *depthwise_bias_diff = bias_diff.data<scalar_t>();
                ConvBackwardBias<scalar_t><<<GET_BLOCKS(count_bias), THREADS_PER_BLOCK>>>(count_bias, top_diff, batch_size,
                    channels, input_height, input_width, conved_height, conved_width, kernel_h, kernel_w, stride_h,
                    stride_w, pad_h, pad_w, depthwise_bias_diff);
            }

            ConvBackwardWeight<scalar_t><<<GET_BLOCKS(count_weight), THREADS_PER_BLOCK>>>(count_weight, top_diff, batch_size,
                channels, input_height, input_width, conved_height, conved_width, kernel_h, kernel_w, stride_h,
                stride_w, pad_h, pad_w, depthwise_weight_diff, bottom_data);

            ConvBackward<scalar_t><<<GET_BLOCKS(count_input), THREADS_PER_BLOCK>>>(count_input, top_diff, batch_size,
                channels, input_height, input_width, conved_height, conved_width, kernel_h, kernel_w, stride_h,
                stride_w, pad_h, pad_w, depthwise_bottom_diff, depthwise_weight);

        }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    if (bias_term_){
        return {bottom_diff, weight_diff, bias_diff};
    }
    else{
        return {bottom_diff, weight_diff};
    }
}
