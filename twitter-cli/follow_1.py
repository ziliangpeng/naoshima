import tweepy
from auth import api
import time
from datetime import datetime, timedelta
import followed


"""
Follows the lists that I (or any user) belong to.
"""

following_ids = map(int, tweepy.Cursor(api.friends_ids, count=2000).items())
# following_ids = map(lambda x: x.id, followers)
print "found", len(following_ids), "following"
print following_ids

name = sys.argv[1]
print 'username is', name
for l in tweepy.Cursor(api.lists_memberships, screen_name=name).items():  # lists a user is added to
    print "List:", l.name
    c = raw_input('follow?:')
    if c.strip() != 'y':
        continue

    list_id = l.id
    i = 0
    for user in tweepy.Cursor(api.list_members, list_id=list_id).items():
        try:
            if int(user.id) in following_ids:
                print "already following", user.id, user.name
                continue
            if user.protected:
                print "User", user.id, user.name, "is protected"
                continue
            if followed.is_followed(user.id):
                print "already followed", user.id, user.name, "before"
                continue
            last_tweeted = user.status.created_at
            if last_tweeted < datetime.now() - timedelta(days = 3):
                print 'User %s last tweeted at %s, is too old' % (user.name, last_tweeted)
            else:
                print datetime.now(), "To follow user:", i, user.id, user.name
                i += 1

                api.create_friendship(user.id)

            if i >= 989:
                print "Already used up today's quote. break."
                break
        except Exception as e:
            print e
        time.sleep(1)
