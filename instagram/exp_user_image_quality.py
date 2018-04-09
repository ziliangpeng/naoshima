import sys
import data
import user_utils


def quality(uname):
    posts = user_utils.get_recent_posts(uname)
    dim = []
    for p in posts:
        h = p['node']['dimensions']['height']
        w = p['node']['dimensions']['width']
        dim.append(max(h, w))
    return sum(dim) / len(dim)

u = sys.argv[1]
for uid in data.get_followed_back(u):
    uname = data.get_id_to_name(uid)
    print(uid, quality(uname))


