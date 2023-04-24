from importlib import import_module

def get(model_name: str, name: str, prompt_template: str):
    module = import_module(f'.{model_name}')
    return module.Model(name, prompt_template)
    
def get_chained(model_name: str):
    module = import_module(f'.{model_name}')
    return module.get_model_for_chain()
