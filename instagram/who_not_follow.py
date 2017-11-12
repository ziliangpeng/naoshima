import auth
import utils

bot = auth.auth()
follows = utils.get_follows(bot)
followers = utils.get_followers(bot)

for id in follows.keys():
    if id not in followers:
        print id, 'https://www.instagram.com/%s/' % (follows[id])
