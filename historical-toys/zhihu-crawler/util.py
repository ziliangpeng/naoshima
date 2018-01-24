import requests
import re


def load_cookies():
    """ Quick hack cookie parsing."""
    COOKIE_FILENAME = 'cookies.secret'
    lines = open(COOKIE_FILENAME).read().split(';')
    cookies = {}
    for line in lines:
        line = line.strip()
        pos = line.index('=')
        k, v = line[:pos], line[pos+1:]
        cookies[k] = v

    return cookies


def follow_user(username, cookies):
    def get_hash_id():
        print 'user', username
        r = requests.get('http://www.zhihu.com/people/%s' % username, cookies=cookies)

        HASH_ID_PATTERN = r'(?<=data\-id\=\")[0-9a-z]{32}'
        return re.findall(HASH_ID_PATTERN, r.text)[0]
    """
        remote action url: http://www.zhihu.com/node/MemberFollowBaseV2
        method: POST
    """

    url = 'http://www.zhihu.com/node/MemberFollowBaseV2'
    hash_id = get_hash_id()
    data = {'method':'follow_member',
            'params':'{"hash_id":"%s"}' % hash_id,
            '_xsrf': cookies['_xsrf']}

    print 'hash id', hash_id
    response = requests.post(url, cookies=cookies, data=data)

    print response.text


