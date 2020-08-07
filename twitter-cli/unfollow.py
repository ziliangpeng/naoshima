import tweepy
import gflags
import time
from datetime import datetime, timedelta
from auth import api
import logging
import os
import sys
import glog

gflags.DEFINE_bool('dryrun', True, '')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)


formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

fh = logging.FileHandler(os.path.realpath('./unfollowed.local'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger = logging.getLogger('TwitterUnfollow')
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)

u = api.get_user(screen_name='recklessdesuka')
my_id = u.id

def main():
    sz = 2000
    for i in reversed(list(tweepy.Cursor(api.friends_ids, count=sz).items())[:sz]):
    # for i in reversed(api.friends_ids()[:500]):
        try:
            u = api.get_user(i)
        except:
            glog.error("error getting user")
            continue
        # logger.info('inspecting %d %s' % (i, u.screen_name))
        if not pass_filter(i, u):
            logger.info('Will destroy %d %s %s' % (i, u.screen_name, u.name))
            if not FLAGS.dryrun:
                logger.info('!!!!! ------ destroying %d %s %s' % (i, u.screen_name, u.name))
                api.destroy_friendship(i)
                #break
        sleep_time = 5
        if not FLAGS.dryrun:
            sleep_time = 240
        time.sleep(sleep_time)


def pass_filter(i, u):
    # b = False
    # b |= follow_me(i)
    # # b |= many_followers(u)
    # # b |= not_too_many_friends(u)
    # b |= is_protected(u)

    if is_protected(u):
        return True
    if not active(u):
        return False
    if follow_me(i):
        return True
    if big_v(u):
        return False

    return False

def big_v(u):
    if u.verified:
        return True
    if u.friends_count == 0:
        return True
    if u.followers_count / u.friends_count > 64:
        return True

def active(u):
    try:
        last_tweeted = u.status.created_at
        if last_tweeted < datetime.now() - timedelta(days=90):
            return False
        return True
    except:
        return False


def many_followers(u):
    return u.followers_count > 100

def not_too_many_friends(u):
    return u.friends_count / u.followers_count < 3

def is_protected(u):
    return u.protected

def follow_me(i):
    friendship = api.show_friendship(source_id=i, target_id=my_id)[0]
    return friendship.following

if __name__ == '__main__':
    main()
