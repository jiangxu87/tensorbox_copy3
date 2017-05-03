from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Tensorbox app',
  ext_modules = cythonize(
       "stitch_wrapper.pyx",
       sources=["stitch_rects.cpp", "./hungarian/hungarian.cpp"],
       language="c++",),
)
