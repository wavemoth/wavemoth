#------------------------------------------------------------------------------
# Copyright (c) 2011, Dag Sverre Seljebotn
# All rights reserved. See LICENSE.txt.
#------------------------------------------------------------------------------

top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_c')
    opt.load('compiler_fc')
    opt.load('python')
    opt.load('inplace', tooldir='tools')
    opt.add_option('--with-libpsht', help='path to libpsht to use for benchmark comparison '
                   '(NOTE: must be built with -fPIC)')

def configure(conf):
    conf.add_os_flags('PATH')
    conf.add_os_flags('PYTHON')
    conf.add_os_flags('PYTHONPATH')
    conf.add_os_flags('FWRAP')
    conf.add_os_flags('INCLUDES')
    conf.add_os_flags('LIB')
    conf.add_os_flags('LIBPATH')
    conf.add_os_flags('STLIB')
    conf.add_os_flags('STLIBPATH')
    conf.add_os_flags('FWRAPFLAGS')
    conf.add_os_flags('CFLAGS')

    if not conf.env.CFLAGS:
        raise RuntimeError("Set CFLAGS while developing")
    conf.load('compiler_c')
    conf.load('compiler_fc')
    conf.check_fortran()
    conf.check_fortran_verbose_flag()
    conf.check_fortran_clib()

    conf.load('python')
    conf.check_python_version((2,5))
    conf.check_python_headers()

    conf.check_tool('numpy', tooldir='tools')
    conf.check_numpy_version(minver=(1,3))
    conf.check_tool('cython', tooldir='tools')
    conf.check_cython_version(minver=(0,11,1))
    conf.check_tool('inplace', tooldir='tools')

    conf.check_libpsht()

    conf.env.LIB_BLAS = ['goto2', 'gfortran']
    conf.env.LIBPATH_BLAS = conf.env.RPATH_BLAS = ['/home/dagss/code/GotoBLAS2']

    conf.env.LIB_MKLBLAS = 'mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m'.split()
    conf.env.LIBPATH_MKLBLAS = conf.env.RPATH_MKLBLAS = ['/opt/intel/mkl/lib/intel64']

    conf.env.LIB_FFTW3 = ['fftw3']

    conf.env.LIB_RT = ['rt']

    conf.env.LIB_MKL = ['mkl_rt']

    conf.env.LIB_PROFILER = ['profiler']
    
#    conf.env.LIBPATH_MKL = ['/opt/intel/mkl/lib/intel64']
#    conf.env.INCLUDES_MKL = ['/opt/intel/mkl/include']

def build(bld):
    bld(source=(['spherew/legendre.pyx'] +
                bld.srcnode.ant_glob(incl=['libpshtlight/*.c'])),
        includes=['libpshtlight'],
        target='legendre',
        use='NUMPY',
        features='c pyext cshlib')

    bld(source=(['spherew/interpolative_decomposition.pyx'] +
                bld.srcnode.ant_glob(incl=['libidlight/*.f'])),
        target='interpolative_decomposition',
        use='fcshlib NUMPY',
        features='fc c pyext cshlib')

    bld(source=(['spherew/matvec.pyx', 'src/matvec.c']),
        includes=['src'],
        target='matvec',
        use='NUMPY',
        features='c pyext cshlib')

    bld(source=(['spherew/butterfly.pyx', 'src/butterfly.c']),
        includes=['src'],
        target='butterfly',
        use='NUMPY BLAS fcshlib',
        features='c pyext cshlib')

    bld(source=(['spherew/fastsht.pyx', 'src/fastsht.c', 'src/butterfly.c',
                 'src/fmm1d.c']),
        includes=['src'],
        target='fastsht',
        use='NUMPY BLAS FFTW3', # PS collision between MKL and FFTW..
        features='c fc pyext cshlib')

    bld(source=(['spherew/psht.pyx']),
        target='psht',
        use='NUMPY PSHT',
        features='c pyext cshlib')

    bld(source=(['spherew/fmm.pyx', 'src/fmm1d.c']),
        target='fmm',
        includes=['src'],
        use='NUMPY MKL',
        features='c pyext cshlib')

    bld(source=(['bench/shbench.c', 'src/fastsht.c', 'src/butterfly.c', 'src/fmm1d.c']),
        includes=['src'],
        install_path='bin',
        target='shbench',
#        use='MKLBLAS RT',
        use='BLAS FFTW3 RT PROFILER',
        features='cprogram c')


from waflib.Configure import conf
from os.path import join as pjoin

@conf
def check_libpsht(conf):
    """
    Settings for libpsht
    """
    prefix = conf.options.with_libpsht
    conf.env.LIB_PSHT = ['psht', 'fftpack', 'c_utils']
    conf.env.LINKFLAGS_PSHT = ['-fopenmp']
    conf.env.LIBPATH_PSHT = [pjoin(prefix, 'lib')]
    conf.env.INCLUDES_PSHT = [pjoin(prefix, 'include')]
    

# vim:ft=python
