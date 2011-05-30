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

    conf.env.LIB_BLAS = 'mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m'.split()
    conf.env.LIBPATH_BLAS = conf.env.RPATH_BLAS = ['/opt/intel/mkl/lib/intel64']

    conf.env.LIB_PERFTOOLS = ['profiler']

    conf.env.LIB_FFTW3 = ['fftw3']

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
        use='NUMPY BLAS',
        features='c pyext cshlib')

    bld(source=(['spherew/fastsht.pyx', 'src/fastsht.c', 'src/butterfly.c']),
        includes=['src'],
        target='fastsht',
        use='NUMPY BLAS', # PS collision between MKL and FFTW..
        features='c pyext cshlib')

    bld(source=(['bench/matmulbench.c']),
        includes=['src'],
        target='matmulbench',
        install_path='bin',
        libs=['math'],
        use='LIBC BLAS',
        features='c cprogram')



# vim:ft=python
