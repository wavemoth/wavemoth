# NumPy tool for waf
from waflib.Configure import conf

@conf
def check_numpy_version(conf, minver, maxver=None):
    conf.start_msg("Checking numpy version")
    minver = tuple(minver)
    if maxver: maxver = tuple(maxver)
    (np_ver_str,) = conf.get_python_variables(
            ['numpy.version.short_version'], ['import numpy'])
    np_ver = tuple([int(x) for x in np_ver_str.split('.')])
    if np_ver < minver or (maxver and np_ver > maxver):
        conf.end_msg(False)
        conf.fatal("numpy version %s is not in the "
                "range of supported versions: minimum=%s, maximum=%s" % (np_ver_str, minver, maxver))
    conf.end_msg(str(np_ver))

@conf
def get_numpy_includes(conf):
    conf.start_msg("Checking numpy includes")
    (np_includes,) = conf.get_python_variables(
            ['numpy.get_include()'], ['import numpy'])
    conf.env.INCLUDES_NUMPY = np_includes
    conf.end_msg('ok (%s)' % np_includes)


def configure(conf):
    conf.check_python_module('numpy')
    conf.get_numpy_includes()
