# -- setup.py --
import numpy as np
from os.path import join as pjoin

from setuptools import setup
from Cython.Distutils import build_ext
from numpy.distutils.extension import Extension

# Define extensions
EXTS = []
for modulename, other_sources, language in (
    ('brainsearch.imagespeed', [], 'c'),
    ):
    pyx_src = pjoin(*modulename.split('.')) + '.pyx'
    EXTS.append(Extension(modulename, [pyx_src] + other_sources,
                          language=language,
                          include_dirs=[np.get_include(), "src"]))

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=EXTS
)
