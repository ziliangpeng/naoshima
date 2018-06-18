import data
from utils import rate_limit_get
# import logs

# logger = logs.logger


def profile_hash(s):
    return '9ca88e465c3f866a76f7adee3871bdd8'

    # tbh not too sure what this gives me
    js_url = 'https://www.instagram.com/static/bundles/base/ProfilePageContainer.js/d735e6d96a3e.js'
    response = rate_limit_get(s, js_url)
    if response.status_code != 200:
        raise Exception(str(response.status_code))
    text = response.text
    index1 = text.find('m="')
    index2 = text.find('m="', index1 + 1) + 3
    index2_end = text.find('"', index2 + 1)
    hash = text[index2:index2_end]
    # logger.info("profile hash is " + hash)
    return hash


def following_hash(s):
    return '58712303d941c6855d4e888c5f0cd22f'

    js_url = 'https://www.instagram.com/static/bundles/base/Consumer.js/ee18369e407b.js'
    response = rate_limit_get(s, js_url)
    if response.status_code != 200:
        raise Exception(str(response.status_code))
    text = response.text
    index1 = text.find('l="')
    index2 = text.find('l="', index1 + 1) + 3
    index3 = text.find('l="', index2 + 1) + 3
    index3_end = text.find('"', index3 + 1)
    hash = text[index3:index3_end]
    # logger.info("following hash is " + hash)
    return hash


def follower_hash(s):
    return '37479f2b8209594dde7facb0d904896a'

    js_url = 'https://www.instagram.com/static/bundles/base/Consumer.js/ee18369e407b.js'
    response = rate_limit_get(s, js_url)
    if response.status_code != 200:
        raise Exception(str(response.status_code))
    text = response.text
    index1 = text.find('),s="') + 5
    index1_end = text.find('"', index1 + 1)
    hash = text[index1:index1_end]
    # logger.info("follower hash is " + hash)
    return hash


if __name__ == '__main__':
    name = 'airbnb.office'
    s = data.get_session(name)
    profile_hash(s)
    following_hash(s)
    follower_hash(s)
