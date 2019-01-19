from distutils.core import setup
from Cython.Build import cythonize

setup(name='_warping',
      ext_modules=cythonize("_warping.pyx"))
