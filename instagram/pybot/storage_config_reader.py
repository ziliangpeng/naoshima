import json5
from utils import _json_path

m = __import__(__name__)
METHODS = {
    'redis_host': (["redis", "host"], None),
    'redis_port': (["redis", "port"], None),
    'redis_cache_host': (["redis-cache", "host"], None),
    'redis_cache_port': (["redis-cache", "port"], None),
    'mysql_host': (["mysql", "host"], None),
    'mysql_port': (["mysql", "port"], None),
}

# TODO: code here is duplicate. refactor and DRY it
for k, (paths, default) in METHODS.items():
    def make_f(_paths, _default):
        return lambda: _load(_paths, _default)

    setattr(m, 'load_' + k, make_f(paths, default))


def _load(paths, default=None):
    with open('storage_config.local', 'r') as f:
        j = json5.loads(f.read())
        return _json_path(j, paths) or default


if __name__ == '__main__':
    print(load_redis_host())
    print(load_redis_port())
