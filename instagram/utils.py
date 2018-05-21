"""
This is the most basic lib to be impored. Do not import other libs here.
"""
from typing import List
from retrying import retry


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


@retry(wait_exponential_max=1000*3600*24*1.5)
def rate_limit_get(s, url):
    response = s.get(url)
    if response.status_code == 429:
        from dd import m
        m.ratelimit_exceeded()
        raise RateLimitedException(str(response.status_code))
    return response

class RateLimitedException(BaseException):
    pass
