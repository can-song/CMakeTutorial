# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# import os
# import os.path as osp

# script_dir = osp.dirname(osp.realpath(__file__))

# src_root = osp.join(script_dir, 'core/op')
# cpp_src = ['my_conv.cpp']

# if __name__ == '__main__':
#     include_dirs = [osp.join(script_dir, 'core/op')]
#     cpp_path = [osp.join(src_root, src) for src in cpp_src]

#     setup(
#         name='conv',
#         ext_modules=[CppExtension('my_ops', cpp_path, include_dirs=include_dirs)],
#         cmdclass={'build_ext': BuildExtension})

from setuptools import setup
from torch.utils import cpp_extension
import os
import os.path as osp



src_root = 'core/op'
cpp_src = ['my_conv.cpp', 'my_conv_cuda.cu']

if __name__ == '__main__':
    include_dirs = ['core/op']
    cpp_path = [osp.join(src_root, src) for src in cpp_src]

    setup(
        name='panoflow',
        ext_modules=[
            cpp_extension.CUDAExtension(
                'my_ops', cpp_path, include_dirs=include_dirs,
                define_macros=[('INDEX_TYPE', 'short')])
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
