from typing import List


def _check_not_null(o):
    if o is None:
        raise BaseException("object cannot be None")
    return o


def _json_path(j: str, paths: List[str]):
    _check_not_null(j)
    for k in paths:
        if k in j:
            j = j[k]
        else:
            return None
    return j

