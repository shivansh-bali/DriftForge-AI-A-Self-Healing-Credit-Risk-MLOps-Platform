import json

REGISTRY_PATH = "models/registry.json"

def get_next_version():

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    next_version = registry["latest_version"] + 1

    return next_version


def update_registry(new_version):

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    registry["latest_version"] = new_version
    registry["production_model"] = f"model_v{new_version}.pkl"

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)


def load_production_model():

    import joblib
    import json

    with open("models/registry.json") as f:
        registry = json.load(f)

    model_path = f"models/{registry['production_model']}"

    return joblib.load(model_path)
