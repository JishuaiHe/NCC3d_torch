from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))
# os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5+PTX'


setup(
    name='Ncc3d_torch',
    packages=find_packages(),
    version='0.0.1',
    author='Jishuai He',
    ext_modules=[
        CUDAExtension(name='Ncc3d', 
                      sources=[
                               "src/kernel/ncc_cuda_naive.cpp",
                               "src/kernel/ncc_cuda_naive_kernel.cu"
                               ],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2', '--extended-lambda']},
                      # TORCH_CUDA_ARCH_LIST="7.5 +PTX",
                      ),
                ],

    cmdclass={'build_ext': BuildExtension}
      
      )