import os
import tempita
import re

def instantiate_template(name, **args):
    path = os.path.join(os.path.dirname(__file__), name) 
    with file(path) as f:
        template = f.read()
    code = tempita.sub(template, **args)
    # Strip comments
    code, n = re.subn(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Strip empty lines
    lines = [line for line in code.split('\n') if len(line.strip()) > 0]
    code = '\n'.join(lines)
    # Output processed file for debugging
    with file(path[:-len('.in')], 'w') as f:
        f.write(code)
    return code
