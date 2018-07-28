import tweepy
from auth import api
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--lang', '-l', help='tweet language')
parser.add_argument('--keyword', '-k', help='keyword to search', required=True)
parser.add_argument('--list', help='list to add to', required=True)
args = parser.parse_args()

keyword = args.keyword
list_id = args.list
lang = args.lang

print('keyword', keyword)
print('list', list_id)
print('lang', lang)

results = api.search(q=keyword, lang=lang)
for r in results:
    print(r.text)
    u = r.user
    print(u.name)
    print('+' * 42)
    api.add_list_member(user_id=u.id, list_id=list_id)
