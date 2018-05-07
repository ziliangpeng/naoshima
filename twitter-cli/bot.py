import tweepy
import time
import sys
from datetime import datetime, timedelta
import random
from auth import api
from threading import Thread
import logging
import followed


"""
THIS IS ONLY A PROTOTYPE.
"""

formatter = logging.Formatter(
    '[%(asctime)s] - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

logger = logging.getLogger('TwitterBotPrototype')
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)


class ForeverThread(Thread):
    def run(self):
        while True:
            try:
                self.act()
            except BaseException as e:
                logger.error(e)
                logger.info("sleeping...")
                time.sleep(60 * 5)

class Unfo(ForeverThread):
    def act(self):
        for friend_id in list(tweepy.Cursor(api.friends_ids, count=5000).items()):
            u = api.get_user(friend_id)
            logger.info("Unfollow: %s(%s)(%d)" % (u.screen_name, u.name, u.id))
            api.destroy_friendship(u.id)
            two_minutes = 60 * 3
            logger.info("Sleep for %d" % two_minutes)
            time.sleep(two_minutes)



class Fo(ForeverThread):
    def filter(self, u):
        friends_count = u.friends_count
        followers_count = u.followers_count
        if u.protected:
            return False
        if followers_count > friends_count:
            return False
        if friends_count > 1000:
            return False
        if u.status:
            last_tweeted = u.status.created_at
            if last_tweeted < datetime.now() - timedelta(days=3):
                return False
        if followed.is_followed(u.id):
            return False
        return True

    def act(self):
        for follower_id in random.sample(list(tweepy.Cursor(api.followers_ids, count=5000).items()), 100):
            for l in tweepy.Cursor(api.lists_memberships, user_id=follower_id).items():  # lists a user is added to
                remain_to_fo_from_list = 64
                for user in tweepy.Cursor(api.list_members, list_id=l.id).items():
                    if remain_to_fo_from_list <= 0:
                        break
                    logger.info("Inspect %s(%s)(%d)?" %(user.screen_name, user.name, user.id))
                    if not self.filter(user):
                        logger.info("Not pass")
                        # time.sleep(3)
                        continue
                    else:
                        logger.info("Follow https://twitter.com/%s (%s)(%d)?" %(user.screen_name, user.name, user.id))
                    api.create_friendship(user.id)
                    followed.follow(user.id)
                    remain_to_fo_from_list -= 1
                    two_minutes = 60 * 4
                    logger.info("Sleep for %d" % two_minutes)
                    time.sleep(two_minutes)


def main():
    if 'fo' in sys.argv:
        Fo().start()
    if 'unfo' in sys.argv:
        Unfo().start()


if __name__ == '__main__':
    main()
