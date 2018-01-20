def register_loaders(methods, method_prefix, loader):
    m = __import__(__name__)
    for k, (paths, default) in methods.items():
        def make_f(_paths, _default):
            return lambda: loader(_paths, _default)

        setattr(m, method_prefix + k, make_f(paths, default))
