_ROUTES = {}


def register(name, model_id):
    _ROUTES[name] = model_id


def get_route(name):
    if name not in _ROUTES:
        raise KeyError(
            f"No recognition route registered for label '{name}'. "
            f"Registered routes: {sorted(_ROUTES)}"
        )
    return _ROUTES[name]


def registered_routes():
    return sorted(_ROUTES)
