import tweepy
from auth import api
import sys



keyword = sys.argv[1]
list_id = sys.argv[2]

results = api.search(q=keyword, lang='zh')
for r in results:
    print(r.text)
    u = r.user
    print(u.name)
    print('+' * 42)
    api.add_list_member(user_id=u.id, list_id=list_id)
