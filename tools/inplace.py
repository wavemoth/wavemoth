"""
waf tool for supporting in-place install, in particular useful for
Python projects.

By passing --inplace to 'waf configure', the behaviour of 'waf install'
is changed:

 - Python extensions are dropped directly to the source
   tree.
 - Other shared libraries are put in a 'lib' directory in the project dir
 - rpath is set up for all shared libraries to contain the 'lib' directory
   in the project dir, using the ${ORIGIN} feature of modern ld.so.

BUGS:

This may only work on Linux? Should probe for success of ${ORIGIN} in
rpath and use an absolute path otherwise. And I have no idea about
Windows.
"""

import os
from waflib.Configure import conf
from waflib.TaskGen import after_method, before_method, feature, taskgen_method, extension

@feature('cshlib', 'fcshlib', 'pyext')
@after_method('process_source')
@before_method('propagate_uselib_vars', 'apply_link', 'init_pyext')
def apply_inplace_install_path_shlib(self):
    if self.env['INPLACE_INSTALL'] and not getattr(self, 'install_path', None):
        if 'pyext' in self.features:
            # Scan sources for likely position of extension source
            # For now, just take Cython source -- not sure how to deal with C extensions
            # which can have multiple C suffixes
            package_node = None
            for x in self.source:
                if x.suffix() == '.pyx':
                    package_node = x.parent
                    break
            else:
                raise AssertionError("Python extension does not have an associated .pyx file")
            self.install_path = package_node.get_src().abspath()
            
            lib_path  = self.bld.srcnode.make_node('lib')
            if not getattr(self, 'rpath', None):
                self.rpath = os.path.join('${ORIGIN}', lib_path.path_from(package_node))
        else:
            self.install_path = self.bld.srcnode.make_node('lib').abspath()
            if not getattr(self, 'rpath', None):
                self.rpath = '${ORIGIN}'

@feature('cprogram')
@before_method('process_source')
def apply_inplace_install_path_cprogram(self):
    self.install_path = self.bld.srcnode.make_node('bin').abspath()
    self.rpath = '${ORIGIN}/../lib'

def options(self):
    self.add_option('--inplace', action='store_true',
                    help='"install" command installs to the project directory')

def configure(self):
    if self.options.inplace:
        self.env['INPLACE_INSTALL'] = True

