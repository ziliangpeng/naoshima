import data
import time
import datetime
import auth
import secret_reader
import user_utils
import random
import requests
from lib.postbot import InstagramAPI


"""
graphql url:
https://www.instagram.com/graphql/query/?query_id=17885113105037631&variables=%7B%22id%22%3A%226575470602%22%2C%22first%22%3A12%2C%22after%22%3A%22AQDiSrivWlEL4I0UzW00KCMhigZwMiPWpYimS2kihXKea8vIRfRaQR40RyTYA0BLHUxaolww2Li3-omro1cwi7kgoM8G5IK3925HrphwyawHFg%22%7D
"""

ACCEPT_TYPES = ["GraphImage", "GraphSidecar"]

def by_url(bot, u):
    my_url = 'https://www.instagram.com/%s/?__a=1' % u

    # get my saved
    time.sleep(30)
    r = bot.s.get(my_url)
    j = r.json()
    saved = j["user"]["saved_media"]["nodes"]
    # random.shuffle(saved)
    print(len(saved), 'saved')
    saved = [s for s in saved if s["__typename"] in ACCEPT_TYPES]
    # other types include GraphSidecar and GraphVideo
    print(len(saved), 'accept type')
    saved = [s for s in saved if not data.is_posted(u, s["id"])]
    print(len(saved), 'not posted')


    # choose one to post
    # chosen = random.choice(saved)
    chosen = saved[-1] # always use the oldest in backlog
    src = chosen["display_src"]
    print('src', src)
    photo_code = chosen["code"]
    photo_id = chosen["id"]
    print('photo code', photo_code)
    photo_url = 'https://www.instagram.com/p/%s/?__a=1' % photo_code
    r = bot.s.get(photo_url)
    j = r.json()
    owner = j["graphql"]["shortcode_media"]["owner"]["username"]
    print('owner', owner)
    caption = chosen["caption"] or ""
    caption = 'by %s \n' % owner + \
              'https://www.instagram.com/p/%s/' % photo_code + \
              '\n' + \
              caption
    caption = caption.replace('@', '')  # do not let anyone know
    print(caption)
    return src, caption, photo_id, photo_code


def by_graphql(bot, u):
    uid = user_utils.get_user_id(u)
    saved = user_utils.get_saved_medias(bot, uid)
    print(len(saved), 'saved')
    saved = [s for s in saved if s.typename in ACCEPT_TYPES]
    print(len(saved), 'accept type')
    saved = [s for s in saved if not data.is_posted(u, s.photo_id)]
    print(len(saved), 'not posted')
    pass
    chosen = saved[-1] # always use the oldest in backlog
    src = chosen.url
    print('src', src)
    photo_code = chosen.code
    photo_id = chosen.photo_id
    print('photo code', photo_code)
    photo_url = 'https://www.instagram.com/p/%s/?__a=1' % photo_code
    r = bot.s.get(photo_url)
    j = r.json()
    owner = j["graphql"]["shortcode_media"]["owner"]["username"]
    print('owner', owner)
    caption = chosen.caption
    caption = 'by @%s \n' % owner + \
              'https://www.instagram.com/p/%s/' % photo_code + \
              '\n' + \
              caption
    caption = caption.replace('@', '')  # do not let anyone know
    print(caption)
    return src, caption, photo_id, photo_code



if __name__ == '__main__':
    # print separator
    print('\n\n' + '<' * 42)
    print(datetime.datetime.now())

    # login
    bot = auth.auth()
    u = secret_reader.load_secrets()[0]

    # src, caption, photo_id, photo_code = by_url(bot, u)
    src, caption, photo_id, photo_code = by_graphql(bot, u)

    # download file
    FILEPATH = '/data/ig_tmp.jpg'
    r = requests.get(src)
    with open(FILEPATH, 'wb') as f:
        f.write(r.content)

    # upload file
    time.sleep(30)
    print('posting...')
    user, pwd = secret_reader.load_secrets()
    InstagramAPI = InstagramAPI(user, pwd)
    InstagramAPI.login() # login

    time.sleep(60)
    success = InstagramAPI.uploadPhoto(FILEPATH, caption=caption)
    if success:
        data.set_posted(u, photo_id, photo_code)