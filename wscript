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
    opt.add_option('--with-acml-lib', help='path to ACML libs to use')
    opt.add_option('--with-perftools', help='path to google-perftools'
                   '(NOTE: must be configured with PIC)')
    opt.add_option('--with-numa', help='path to NUMA')
    opt.add_option('--patched-libpsht', action='store_true',
                   help='libpsht is patched to enable selective benchmarks')
    opt.add_option('--no-openmp', action='store_true')
    opt.add_option('--butterfly-only', action='store_true')

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
    conf.add_os_flags('LINKFLAGS')

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
    if not conf.options.butterfly_only:
        conf.check_libpsht()
        conf.check_fftw3()
        conf.check_google_perftools()
        conf.check_blas()
        conf.check_numa()
    conf.env.BUTTERFLY_ONLY = conf.options.butterfly_only

    conf.env.LIB_RT = ['rt']
    conf.env.LIB_MATH = ['m']
    conf.env.LIB_MKL = ['mkl_rt']

    conf.env.CFLAGS_PROFILEGEN = ['-fprofile-generate']
    conf.env.LINKFLAGS_PROFILEGEN = ['-fprofile-generate']
    conf.env.CFLAGS_PROFILEUSE = ['-fprofile-use']
    conf.env.LINKFLAGS_PROFILEUSE = ['-fprofile-use']

    conf.env.CFLAGS_C99 = ['-std=gnu99']
    conf.env.CYTHONFLAGS = ['-a']

    if not conf.options.no_openmp:
        conf.env.CFLAGS_OPENMP = ['-fopenmp']
        conf.env.LINKFLAGS_OPENMP = ['-fopenmp']
    
#    conf.env.LIBPATH_MKL = ['/opt/intel/mkl/lib/intel64']
#    conf.env.INCLUDES_MKL = ['/opt/intel/mkl/include']

def build(bld):
    #
    # Main shared library
    #
    bld(target='src/butterfly.h',
        source=['src/butterfly.h.in'],
        rule=run_tempita)

    if bld.env.BUTTERFLY_ONLY:
        bld(target='fastsht',
            source=['src/butterfly.c.in'],
            includes=['src'],
            use='C99',
            features='c cshlib')
    else:
        bld(target='fastsht',
            source=['src/fastsht.c', 'src/butterfly.c.in', 'src/legendre_transform.c.in'],
            includes=['src'],
            use='C99 BLAS FFTW3 OPENMP NUMA',
            features='c cshlib')

    bld.add_manual_dependency(
        bld.path.find_resource('src/butterfly.c.in'),
        bld.path.find_resource('src/butterfly.h.in'))

    #
    # Python wrappers
    #
    
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

    bld(source=(['spherew/butterfly.pyx']),
        includes=['src'],
        target='butterfly',
        use='NUMPY fcshlib fastsht',
        features='c pyext cshlib')

    bld(source=(['spherew/butterflylib.pyx']),
        includes=['src'],
        target='butterflylib',
        use='NUMPY fcshlib fastsht',
        features='c pyext cshlib')

    # NOTE: BUTTERFLY_ONLY RETURNS HERE
    if bld.env.BUTTERFLY_ONLY:
        return

    bld(source=(['spherew/lib.pyx']),
        includes=['src'],
        target='lib',
        use='NUMPY fastsht',
        features='c fc pyext cshlib')
    for x in ['src/fastsht.h', 'src/butterfly.h.in']:
        bld.add_manual_dependency(
            bld.path.find_resource('spherew/lib.pyx'),
            bld.path.find_resource(x))

    bld(source=(['spherew/psht.pyx']),
        target='psht',
        use='NUMPY PSHT',
        features='c pyext cshlib')

    bld(source=(['spherew/_openmp.pyx']),
        target='_openmp',
        use='OPENMP',
        features='c pyext cshlib')

    bld(source=(['spherew/blas.pyx']),
        includes=['src'],
        target='blas',
        use='NUMPY BLAS',
        features='c pyext cshlib')
    bld.add_manual_dependency(
        bld.path.find_resource('spherew/blas.pyx'),
        bld.path.find_resource('src/blas.h'))

    #
    # Standalone C programs
    #

    bld(source=['bench/cpubench.c'],
        includes=['src'],
        target='cpubench',
        use='C99 BLAS OPENMP',
        features='cprogram c')

    bld(source=['bench/numabench.c'],
        includes=['src', 'bench'],
        target='numabench',
        use='C99 RT MATH NUMA',
        features='cprogram c')

    bld(source=['bench/shbench.c'],
        includes=['src'],
        target='shbench',
        use='C99 RT PSHT OPENMP NUMA fastsht',
        features='cprogram c')

    if bld.env.HAS_PERFTOOLS:
        bld(source=(['bench/shbench.c']),
            includes=['src'],
            install_path='bin',
            target='shbench-prof',
            use='C99 RT PSHT OPENMP PERFTOOLS fastsht',
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
    if conf.options.patched_libpsht:
        conf.env.CFLAGS_PSHT = ['-DPATCHED_LIBPSHT=1']
    # Check presence of libpsht in general
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
        use='PSHT')
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
def check_blas(conf):
    """
    Settings for BLAS
    """
    conf.start_msg("Checking for BLAS")
    path = []
    if conf.options.with_acml_lib:
        path = conf.options.with_acml_lib
        conf.env.LIB_BLAS = 'acml acml_mv'.split()
        name = 'ACML'
    else:
        conf.env.LIB_BLAS = 'f77blas atlas gfortran'.split()
        path = conf.options.with_atlas_lib
        name = 'ATLAS'
        
    conf.env.LIBPATH_BLAS = path
    conf.env.RPATH_BLAS = path
            
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
        use='BLAS')
    conf.end_msg('%s at %s' % (name, path if path else '(default path)'))

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
    conf.env.CFLAGS_PERFTOOLS = ['-DHAS_PPROF=1']
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
        conf.env.HAS_PERFTOOLS = False
    else:
        conf.env.LIB_MAYBEPERFTOOLS = conf.env.LIB_PERFTOOLS
        conf.env.CFLAGS_MAYBEPERFTOOLS = conf.env.CFLAGS_PERFTOOLS
        conf.env.HAS_PERFTOOLS = True
    conf.end_msg(prefix if prefix else True)

@conf
def check_numa(conf):
    conf.start_msg("Checking for NUMA")
    conf.env.LIB_NUMA = ['numa', 'pthread']
    if conf.options.with_numa:
        prefix = conf.options.with_numa
        conf.env.RPATH_NUMA = conf.env.LIBPATH_NUMA = pjoin(prefix, 'lib')
        conf.env.INCLUDES_NUMA = pjoin(prefix, 'include')

    cfrag = dedent('''\
    #include <numa.h>
    #if (LIBNUMA_API_VERSION != 2)
    #error Currently only NUMA API version 2 supported
    #endif
    int main() {
	struct bitmask *nodes;
	nodes = numa_allocate_nodemask();
        numa_bitmask_clearall(nodes);
    }
    ''')
    
    conf.check_cc(
        fragment=cfrag,
        features = 'c',
        compile_filename='test.c',
        use='NUMA')

    conf.end_msg(conf.options.with_numa if conf.options.with_numa else '(default path)')

from waflib import TaskGen

def run_tempita(task):
    import tempita
    import re
    assert len(task.inputs) == len(task.outputs) == 1
    tmpl = task.inputs[0].read()
    result = tempita.sub(tmpl)
    result, n = re.subn(r'/\*.*?\*/', '', result, flags=re.DOTALL)
    result = '\n'.join('/*!*/  %s' % x for x in result.splitlines())
    result = '/* DO NOT EDIT THIS FILE, IT IS GENERATED */\n%s' % result
    task.outputs[0].write(result)

TaskGen.declare_chain(
        name = "tempita",
        rule = run_tempita,
        ext_in = ['.c.in'],
        ext_out = ['.c'],
        reentrant = True,
        )

# vim:ft=python
