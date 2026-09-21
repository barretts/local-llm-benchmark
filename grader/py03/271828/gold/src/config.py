from copy import deepcopy

def overlay_config(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            result[key] = overlay_config(base[key], value)
        else:
            result[key] = deepcopy(value)
    return result
