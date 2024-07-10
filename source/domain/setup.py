from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='encode_competences',
    ext_modules=[
        CUDAExtension(
            name='encode_competences',
            sources=['encode_competences_kernel.cu'],
            include_dirs=['C:\\Users\\marco\\anaconda3\\Lib\\site-packages\\sentence_transformers', 
                          'C:\\Users\\marco\\.conda\\envs\\beakerx\\Lib\\site-packages\\sentencepiece\\include', 
                          torch.utils.cpp_extension.include_paths()],
            library_dirs=['C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\lib\\x64'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
