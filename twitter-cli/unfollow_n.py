import tweepy
from auth import api
import sys
import itertools

cnt = int(sys.argv[1])
print('to unfollow %d users', cnt)

for user in itertools.islice(tweepy.Cursor(api.friends).items(), cnt):  # lists all follows
    print(user.screen_name)
    print(user.name)
    print(user.id)
    api.destroy_friendship(user.id)
