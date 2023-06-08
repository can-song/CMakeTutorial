from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

import os
import os.path as osp
import glob

import torch
if not torch.cuda.is_available():
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', "6.1")

script_dir = osp.dirname(osp.realpath(__file__))

def get_version():
    version_file = 'ssconv/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

if __name__ == '__main__':
    include_dirs = [osp.join(script_dir, 'ssconv/core/include')]
    src_dir      = osp.join(script_dir, 'ssconv/core/src')
    sources      = [osp.join(src_dir, 'ssconv.cpp'),
                    osp.join(src_dir, 'ssconv_cuda.cu')]

    # setup(
    #     name='ssconv',
    #     ext_modules=[CppExtension('ssconv', sources, include_dirs=include_dirs)],
    #     cmdclass={'build_ext': BuildExtension})
    setup(
        name='ssconv',
        version=get_version(),
        packages=find_packages(),
        ext_modules=[CUDAExtension('ssconv.core', sources, include_dirs=include_dirs,
                                   define_macros=[("INDEX_TYPE", "short")],
                                   undef_macros=[])],
        cmdclass={'build_ext': BuildExtension})