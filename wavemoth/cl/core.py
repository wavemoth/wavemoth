import os
import tempita

def instantiate_template(name, **args):
    path = os.path.join(os.path.dirname(__file__), name) 
    with file(path) as f:
        template = f.read()
    return tempita.sub(template, **args)
