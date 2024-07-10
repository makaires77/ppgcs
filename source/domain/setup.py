from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='encode_competences',  # Nome do módulo
    ext_modules=[
        CUDAExtension(
            name='encode_competences',  # Nome da extensão
            sources=['encode_competences_kernel.cu'],
            include_dirs=['C:\\Users\\marco\\anaconda3\\Lib\\site-packages\\sentence_transformers', 'C:\\Users\\marco\\.conda\\envs\\beakerx\\Lib\\site-packages\\sentencepiece\\include'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        )
    ]
)