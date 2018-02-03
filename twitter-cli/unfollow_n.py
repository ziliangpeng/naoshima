import tweepy
from auth import api
import sys
import itertools
import logging


formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

fh = logging.FileHandler('/data/twitter/logs/unfollow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger = logging.getLogger('InstagramBot')
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)

cnt = int(sys.argv[1])
logger.info('To unfollow %d users', cnt)

for user in itertools.islice(tweepy.Cursor(api.friends).items(), cnt):  # lists all follows
    logger.info("%d: %s(%s)", user.id, user.screen_name, user.name)
    api.destroy_friendship(user.id)
