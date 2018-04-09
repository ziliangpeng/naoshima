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
    avg = sum(dim) / len(dim)
    maxi = max(dim)
    dim.sort()
    med = dim[len(dim)/2]
    return avg, maxi, med

u = sys.argv[1]
for uid in data.get_followed_back(u):
    try:
        uname = data.get_id_to_name(uid)
        print("https://www.instagram.com/%s/" % uname, quality(uname))
    except BaseException as e:
        print(e)


