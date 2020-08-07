import itertools
import tweepy
from auth import api
import time
import sys
from datetime import datetime, timedelta
import followed
import glog
from langdetect import detect_langs
from retrying import retry
import random


MY_NAME = 'recklessdesuka'


def get_followings():
    following_ids = list(
        map(int, list(tweepy.Cursor(api.friends_ids, count=2000).items())))
    # following_ids = map(lambda x: x.id, followers)
    print("found", len(following_ids), "following")
    return following_ids


def same_list(name=MY_NAME):
    following_ids = get_followings()

    print('username is', name)
    i = 0
    # lists a user is added to
    for l in tweepy.Cursor(api.lists_memberships, screen_name=name).items():
        print("List:", l.name)
        print("size:", l.member_count)
        c = input('follow?:')
        if c.strip() != 'y':
            continue

        list_id = l.id
        MAX_FROM_LIST = 128
        try:
            for user in itertools.islice(tweepy.Cursor(api.list_members, list_id=list_id).items(), MAX_FROM_LIST):
                try:
                    if int(user.id) in following_ids:
                        print("already following", user.id, user.name)
                        continue
                    if user.protected:
                        print("User", user.id, user.name, "is protected")
                        continue
                    if followed.is_followed(user.id):
                        print("already followed", user.id, user.name, "before")
                        continue
                    last_tweeted = user.status.created_at
                    if last_tweeted < datetime.now() - timedelta(days=3):
                        print('User %s last tweeted at %s, is too old' %
                              (user.name, last_tweeted))
                    else:
                        print(datetime.now(), "To follow user:",
                              i, user.id, user.name)
                        i += 1

                        api.create_friendship(user.id)
                        time.sleep(240)

                    if i >= 989:
                        print("Already used up today's quote. break.")
                        break
                except Exception as e:
                    print(e)
                time.sleep(1)
        except Exception as e:
            print(e)


@retry(wait_random_min=10000, wait_random_max=1000 * 60 * 10, stop_max_attempt_number=7)
def get_followers_ids(u=None):
    glog.info("Getting followers for %s" % (str(u)))
    if u:
        return api.followers_ids(user_id=u)
    else:
        return api.followers_ids()


def fofo():

    followers_ids = [115302629]  # 坂本
    # followers_ids = list(
    #     map(int, list(tweepy.Cursor(api.followers_ids, count=5000).items())))
    followers_ids = get_followers_ids()
    # print(dir(api))
    following_ids = get_followings()
    # following_ids = []

    random.shuffle(followers_ids)
    for u_id in followers_ids:
        try:
            fo = api.get_user(u_id)
        except:
            glog.error("Cannot visit user %d" % (u_id))
            continue
        if fo.protected:
            glog.info("User %s is protected. Skip." % (fo.screen_name))
            continue
        glog.info("Inspecting foer %d %s" % (fo.id, fo.screen_name))
        # fofo_ids = [1105041123461025793]
        # fofo_ids = list(
        #     map(int, list(tweepy.Cursor(api.followers_ids, user_id=u_id, count=20).items())))
        try:
            fofo_ids = get_followers_ids(u_id)
        except:
            glog.error("CANNOT GET FO of %s" % (str(u_id)))
            continue
        random.shuffle(fofo_ids)
        for fofo_id in fofo_ids:
            # filter it, and follow it
            try:
                u = api.get_user(fofo_id)
            except:
                glog.error("  Cannot visit fofo user %d" % (fofo_id))
                continue
            if fofo_id in following_ids:
                glog.info("  Already following", u.id, u.name)
                continue
            glog.info("  Analyzing fofoer %d %s" % (u.id, u.screen_name))
            if u.protected:
                glog.info("    User %s is protected. Skip." % (u.screen_name))
                continue
            try:
                tweets = api.user_timeline(user_id=fofo_id)
            except:
                glog.error("  Cannot get timeline for fofo user %d %s" %
                           (fofo_id, u.name))
                continue
            if len(tweets) == 0:
                glog.info("    No tweet.")
                continue
            try:
                last_tweeted = u.status.created_at
            except:
                glog.error("  Cannot get last tweet from fofo user %d %s" % (fofo_id, u.name))
                continue
            if last_tweeted < datetime.now() - timedelta(days=3):
                glog.info("    User %s last tweeted at %s, is too old" %
                          (u.name, last_tweeted))
                continue
            if u.followers_count < 100:
                glog.info("    User %s has <100 followers. Skip." % (u.name))
                continue
            tweet = tweets[0]
            glog.info("    " + tweet.text)
            try:
                langs = detect_langs(str(tweet.text))
            except:
                glog.error('Cannot determine text. skip.')
                continue
            if len(langs) == 0:
                glog.info("    No language detected.")
                continue
            glog.info('    ' + str(langs))
            for lang in langs:
                lang = lang.lang
                if lang in ['zh-cn', 'zh-tw']:
                    glog.info("    中国 user. To follow")
                    try:
                        api.create_friendship(fofo_id)
                    except:
                        glog.error("Cannot follow.")
                        continue
                    time.sleep(240)
                    break
            # else:
            #     glog.info("    lang is %s. not follow" % (lang))
            time.sleep(5)


def main():
    # name = sys.argv[1]
    # same_list(name)
    fofo()


if __name__ == "__main__":
    main()
