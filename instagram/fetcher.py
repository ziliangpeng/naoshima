import json


def get_user_json(u, s=None):
    if s is None:
        import requests
        s = requests
    url = 'https://www.instagram.com/%s' % u
    r = s.get(url)
    if r.status_code != 200:
        return r.status_code, ''
    else:
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
