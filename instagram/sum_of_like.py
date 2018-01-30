import requests
import sys
from user_utils import _json_path
from datadog import statsd

u = sys.argv[1]


url = 'http://www.instagram.com/%s/?__a=1' % u

r = requests.get(url)
if r.status_code == 200:
    j = r.json()
    posts = _json_path(j, ["user", "media", "nodes"])
    likes = [_json_path(p, ["likes", "count"]) for p in posts]
    statsd.gauge('naoshima.ig.likes', sum(likes), tags=["user:" + u])
    print(sum(likes))

    follows = _json_path(j, ["user", "follows", "count"])
    followed_by = _json_path(j, ["user", "followed_by", "count"])
    statsd.gauge('naoshima.ig.stats', sum(likes), tags=["user:" + u, "type:likes"])
    statsd.gauge('naoshima.ig.stats', follows, tags=["user:" + u, "type:follows"])
    statsd.gauge('naoshima.ig.stats', followed_by, tags=["user:" + u, "type:followed_by"])
else:
    pass