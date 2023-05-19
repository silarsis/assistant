from importlib import import_module

def get(model_name: str):
    module = import_module(f'models.{model_name}')
    return module.Model()
    
def get_chained(model_name: str):
    module = import_module(f'models.{model_name}')
    return module.get_model_for_chain()
