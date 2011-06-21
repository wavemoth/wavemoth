#./waf-light --tools=compat15,swig,fc,compiler_fc,fc_config,fc_scan,gfortran,g95,ifort,gccdeps;


from textwrap import dedent

top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_c')
    opt.load('compiler_fc')
    opt.load('python')
    opt.load('inplace', tooldir='tools')
    opt.add_option('--with-libpsht', help='path to libpsht to use for benchmark comparison '
                   '(NOTE: must be built with -fPIC)')
    opt.add_option('--with-fftw3', help='path to FFTW3 to use '
                   '(NOTE: must be configured with --with-pic)')
    opt.add_option('--with-atlas-lib', help='path to ATLAS libs to use '
                   '(NOTE: must be configured with PIC)')
    opt.add_option('--with-perftools', help='path to google-perftools'
                   '(NOTE: must be configured with PIC)')

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

    # Libraries
    conf.check_libpsht()
    conf.check_fftw3()
    conf.check_google_perftools()
    conf.check_atlas()

#    conf.env.LIB_BLAS = ['goto2', 'gfortran']
#    conf.env.LIBPATH_BLAS = conf.env.RPATH_BLAS = ['/home/dagss/code/GotoBLAS2']

#    conf.env.LIB_MKLBLAS = 'mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m'.split()
#    conf.env.LIBPATH_MKLBLAS = conf.env.RPATH_MKLBLAS = ['/opt/intel/mkl/lib/intel64']

    conf.env.LIB_RT = ['rt']
    conf.env.LIB_MKL = ['mkl_rt']

    conf.env.CFLAGS_OPENMP = ['-fopenmp']
    conf.env.LINKFLAGS_OPENMP = ['-fopenmp']
    
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
        use='NUMPY ATLAS fcshlib',
        features='c pyext cshlib')

    bld(source=(['spherew/fastsht.pyx', 'src/fastsht.c', 'src/butterfly.c.in',
                 'src/fmm1d.c']),
        includes=['src'],
        target='fastsht',
        use='NUMPY ATLAS FFTW3',
        features='c fc pyext cshlib')

    bld(source=(['spherew/psht.pyx']),
        target='psht',
        use='NUMPY PSHT',
        features='c pyext cshlib')

    ## bld(source=(['spherew/fmm.pyx', 'src/fmm1d.c']),
    ##     target='fmm',
    ##     includes=['src'],
    ##     use='NUMPY MKL',
    ##     features='c pyext cshlib')

    bld(source=(['bench/shbench.c', 'src/fastsht.c', 'src/butterfly.c.in', 'src/fmm1d.c']),
        includes=['src'],
        install_path='bin',
        target='shbench',
        use='ATLAS FFTW3 RT OPENMP PSHT MAYBEPERFTOOLS',
        features='cprogram c')


from waflib.Configure import conf
from os.path import join as pjoin

@conf
def check_libpsht(conf):
    """
    Settings for libpsht
    """
    conf.start_msg("Checking for libpsht")
    prefix = conf.options.with_libpsht
    if not prefix:
        conf.fatal("--with-libpsht not used (FIXME)")
    conf.env.LIB_PSHT = ['psht', 'fftpack', 'c_utils']
    conf.env.LINKFLAGS_PSHT = ['-fopenmp']
    conf.env.LIBPATH_PSHT = [pjoin(prefix, 'lib')]
    conf.env.INCLUDES_PSHT = [pjoin(prefix, 'include')]
    cfrag = dedent('''\
    #include <psht.h>
    #include <psht_geomhelpers.h>
    psht_alm_info *x;
    psht_geom_info *y;
    pshtd_joblist *z;
    int main() {
    /* Only intended for compilation */
      psht_make_general_alm_info(10, 10, 1, NULL, 0, &x);
      psht_make_healpix_geom_info(4, 1, &y);
      pshtd_make_joblist(&z);
      pshtd_execute_jobs(x, y, z);
      return 0;
    }
    ''')
    conf.check_cc(
        fragment=cfrag,
        features = 'c',
        compile_filename='test.c',
        use='PSHT',
        msg='Checking for libpsht')
    conf.end_msg(prefix if prefix else True)

@conf
def check_fftw3(conf):
    """
    Settings for FFTW3
    """
    conf.start_msg("Checking for FFTW3")
    conf.env.LIB_FFTW3 = ['fftw3']
    prefix = conf.options.with_fftw3
    if prefix:
        conf.env.LIBPATH_FFTW3 = [pjoin(prefix, 'lib')]
        conf.env.INCLUDES_FFTW3 = [pjoin(prefix, 'include')]
    cfrag = dedent('''\
    #include <fftw3.h>
    int main() {
    /* Only intended for compilation */
      fftw_plan plan;
      fftw_plan_dft_c2r_1d(4, NULL, NULL, FFTW_ESTIMATE);
      fftw_execute(plan);
      return 0;
    }
    ''')
    conf.check_cc(
        fragment=cfrag,
        features = 'c',
        compile_filename='test.c',
        use='FFTW3')
    conf.end_msg(prefix if prefix else True)

@conf
def check_atlas(conf):
    """
    Settings for ATLAS
    """
    conf.start_msg("Checking for ATLAS")
    conf.env.LIB_ATLAS = 'f77blas atlas gfortran'.split()
    path = conf.options.with_atlas_lib
    if path:
        conf.env.LIBPATH_ATLAS = path
    cfrag = dedent('''\
    void dgemm_(void); /* just check existence of symbol through compilation*/
    int main() {
      dgemm_();
    }
    ''')
    conf.check_cc(
        fragment=cfrag,
        features = 'c',
        compile_filename='test.c',
        use='ATLAS')
    conf.end_msg(path if path else True)

@conf
def check_google_perftools(conf):
    """
    Settings for Google perftools
    """
    conf.start_msg("Checking for google-perftools")
    cfrag = dedent('''\
    #include <google/profiler.h>
    int main() {
      ProfilerStart("foo");
      ProfilerStop();
    }
    ''')
    conf.env.LIB_PERFTOOLS = ['profiler']
    prefix = conf.options.with_perftools
    if prefix:
        conf.env.LIBPATH_PERFTOOLS = [pjoin(prefix, 'lib')]
        conf.env.INCLUDES_PERFTOOLS = [pjoin(prefix, 'include')]
    try:
        conf.check_cc(
            fragment=cfrag,
            features = 'c',
            compile_filename='test.c',
            use='PERFTOOLS')
    except conf.errors.ConfigurationError:
        conf.env.LIB_MAYBEPERFTOOLS = []
    else:
        conf.env.LIB_MAYBEPERFTOOLS = conf.env.LIB_PERFTOOLS
        conf.env.CFLAGS_MAYBEPERFTOOLS = ['-DHAS_PPROF=1']
    conf.end_msg(prefix if prefix else True)


from waflib import TaskGen

def run_tempita(task):
    import tempita
    assert len(task.inputs) == len(task.outputs) == 1
    tmpl = task.inputs[0].read()
    result = tempita.sub(tmpl)
    task.outputs[0].write(result)

TaskGen.declare_chain(
        name = "tempita",
        rule = run_tempita,
        ext_in = ['.c.in'],
        ext_out = ['.c'],
        reentrant = True,
        )

# vim:ft=python
