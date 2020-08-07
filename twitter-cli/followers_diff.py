import gflags
import tweepy
from auth import api
import sys
import json
from datetime import datetime, timedelta
import followed
import os

gflags.DEFINE_bool('refresh', False, 'get new data')
gflags.DEFINE_bool('all', False, 'get new data')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

FOLLOWERS_DIFF_FILENAME = "followers_diff.local"


def read():
    if not os.path.exists(FOLLOWERS_DIFF_FILENAME):
        return []

    with open(FOLLOWERS_DIFF_FILENAME) as f:
        content = f.read()
        return json.loads(content)


def write(data):
    with open(FOLLOWERS_DIFF_FILENAME, 'w') as f:
        f.write(json.dumps(data, indent=2))


data = read()

if FLAGS.refresh:
    followers_ids = list(
        map(int, list(tweepy.Cursor(api.followers_ids, count=5000).items())))

    data.append({'time': str(datetime.now()), 'ids': followers_ids})

    write(data)

if not FLAGS.all:
    data = data[-2:]

for i in range(len(data) - 1):
    prev = data[i]
    current = data[i+1]
    prev_ids = set(prev['ids'])
    cur_ids = set(current['ids'])
    print('diff between %s and %s' % (prev['time'], current['time']))
    for id in prev_ids - cur_ids:
        try:
            u = api.get_user(id)
            print('- %d %s %s' % (u.id, u.screen_name, u.name))
        except:
            print('- Cannot get user %d' % (id))
    for id in cur_ids - prev_ids:
        try:
            u = api.get_user(id)
            print('+ %d %s %s' % (u.id, u.screen_name, u.name))
        except:
            print('+ Cannot get user %d' % (id))

    print('')
