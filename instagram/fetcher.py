import json
from utils import rate_limit_get
from dd import m


def get_user_json(u, s=None):
    if s is None:
        import requests
        s = requests
    url = 'https://www.instagram.com/%s' % u
    r = rate_limit_get(s, url)
    if r.status_code != 200:
        m.get_profile(False)
        return r.status_code, ''
    else:
        m.get_profile(True)
        html = r.text
        lines = html.split('\n')
        for line in lines:
            if 'window._sharedData' in line:
                data = line
                break

        start = data.index('{')
        end = data.rindex('}')
        json_text = data[start:end + 1]
        j = json.loads(json_text)
        j = j['entry_data']['ProfilePage'][0]
        return 200, j


if __name__ == '__main__':
    get_user_json('instagram')
