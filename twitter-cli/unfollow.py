import tweepy
import time
from auth import api

u = api.get_user(screen_name='recklessdesuka')
my_id = u.id

def main():
    for i in reversed(list(tweepy.Cursor(api.friends_ids, count=5000).items())):
        u = api.get_user(i)
        print('inspecting', i, u.screen_name)
        if not pass_filter(i):
            print('destroying', i)
            #api.destroy_friendship(i)
            #break
        time.sleep(1)


def pass_filter(i):
    return follow_me(i)


def follow_me(i):
    friendship = api.show_friendship(source_id=i, target_id=my_id)[0]
    return friendship.following

if __name__ == '__main__':
    main()
