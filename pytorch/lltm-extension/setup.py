# from setuptools import setup, Extension
# from torch.utils import cpp_extension

# setup(name='lltm_cpp',
#       ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})

# Extension(
#    name='lltm_cpp',
#    sources=['lltm.cpp'],
#    include_dirs=cpp_extension.include_paths(),
#    language='c++')

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm',
    ext_modules=[
        CUDAExtension('lltm_cuda', [
            'lltm_cuda.cpp',
            'lltm_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })