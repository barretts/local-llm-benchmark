def overlay_config(base: dict, overrides: dict) -> dict:
    result = base.copy()
    for key, value in overrides.items():
        if not value:
            continue
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key].update(overlay_config(result[key], value))
        elif isinstance(result.get(key), list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value
    return result
