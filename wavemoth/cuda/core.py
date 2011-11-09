import os
import tempita

def instantiate_template(name, **args):
    path = os.path.join(os.path.dirname(__file__), name) 
    with file(path) as f:
        template = f.read()
    code = tempita.sub(template, **args)
    # Output processed file for debugging
    with file(path[:-len('.in')], 'w') as f:
        f.write(code)
    return code
