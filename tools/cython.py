# Cython tool for waf

from waflib.Configure import conf
from waflib import TaskGen

TaskGen.declare_chain(
        name = "cython",
        rule = "${CYTHON} ${CYTHONFLAGS} ${CPPPATH_ST:INCPATHS} ${SRC} -o ${TGT}",
        ext_in = ['.pyx'],
        ext_out = ['.c'],
        reentrant = True,
        )

@conf
def check_cython_version(conf, minver):
    conf.start_msg("Checking cython version")
    minver = tuple(minver)
    import re
    version_re = re.compile(r'cython\s*version\s*(?P<major>\d*)\.(?P<minor>\d*)(?:\.(?P<micro>\d*))?', re.I).search
    cmd = conf.cmd_to_list(conf.env['CYTHON'])
    cmd = cmd + ['--version']
    from waflib.Tools import fc_config
    stdout, stderr = fc_config.getoutput(conf, cmd)
    if stdout:
        match = version_re(stdout)
    else:
        match = version_re(stderr)
    if not match:
        conf.fatal("cannot determine the Cython version")
    cy_ver = [match.group('major'), match.group('minor')]
    if match.group('micro'):
        cy_ver.append(match.group('micro'))
    else:
        cy_ver.append('0')
    cy_ver = tuple([int(x) for x in cy_ver])
    if cy_ver < minver:
        conf.end_msg(False)
        conf.fatal("cython version %s < %s" % (cy_ver, minver))
    conf.end_msg(str(cy_ver))

def configure(conf):
    conf.find_program('cython', var='CYTHON')

