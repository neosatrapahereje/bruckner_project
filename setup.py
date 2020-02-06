from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions =[
    Extension("distances", ["alignment/distances.pyx"],
              # include_dirs=['./alignment']
    )
    ]

setup(
    ext_modules=cythonize(extensions)
    # ext_modules = cythonize("alignment/distances.pyx")
)





