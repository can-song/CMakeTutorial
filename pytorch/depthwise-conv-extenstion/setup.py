from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depethwise_conv',
    ext_modules=[
        CUDAExtension('depthwise_conv_cuda', [
            './depthwise_conv_cuda.cpp',
            './depthwise_conv_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})