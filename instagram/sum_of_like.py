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
    statsd.gauge('naoshima.ig.likes', sum(likes), tags=["user:"+u])
    print(sum(likes))
else:
    pass
