import requests
import langdetect


def langs(u):
    url = 'https://www.instagram.com/%s/?__a=1' % (u)
    r = requests.get(url)
    j = r.json()
    caps = ["caption" in n and n["caption"] or "" for n in j["user"]["media"]["nodes"]]  # TODO: use user_utils
    try:
        return langdetect.detect_langs(' '.join(caps))
    except:
        print('cannot detect lang for user %s' % u)
        return []

