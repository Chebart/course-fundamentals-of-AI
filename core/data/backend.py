import importlib

AVAILABLE_BACKENDS = {}
for name in ["numpy", "cupy"]:
    try:
        module = importlib.import_module(name)
        AVAILABLE_BACKENDS[name] = module
    except Exception as e:
        raise e
