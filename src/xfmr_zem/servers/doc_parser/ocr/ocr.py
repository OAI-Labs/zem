import yaml

funcs = {}
def register(name):
    def decorator(func):
        funcs[name] = func
        return func
    return decorator

def make(ocr_spec):
    name = ocr_spec["name"]
    config_path = ocr_spec.get("config", None)
    if (config_path is None):
        raise ValueError(f"OCR config path is required for {name}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)

    if (args is None):
        return funcs[name]()
    else:
        return funcs[name](**args)