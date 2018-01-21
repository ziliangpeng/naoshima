import auth
import utils

bot = auth.auth()
follows = utils.get_follows(bot)
followers = list(utils.get_all_followers_gen(bot))

for id in list(follows.keys()):
    if id not in followers:
        print(id, 'https://www.instagram.com/%s/' % (follows[id]))
