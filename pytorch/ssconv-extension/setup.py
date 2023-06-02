from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

import os
import os.path as osp
import glob

script_dir = osp.dirname(osp.realpath(__file__))

if __name__ == '__main__':
    include_dirs = [osp.join(script_dir, 'core')]
    src_dir      = osp.join(script_dir, 'core')
    sources      = [osp.join(src_dir, 'ssconv.cpp'),
                    osp.join(src_dir, 'ssconv.cu')]

    # setup(
    #     name='ssconv',
    #     ext_modules=[CppExtension('ssconv', sources, include_dirs=include_dirs)],
    #     cmdclass={'build_ext': BuildExtension})
    setup(
        name='ssconv',
        ext_modules=[CUDAExtension('ssconv', sources, include_dirs=include_dirs)],
        cmdclass={'build_ext': BuildExtension})