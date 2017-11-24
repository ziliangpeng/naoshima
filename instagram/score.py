import requests

import auth
import secret_reader
import user_utils
import sys
import time
import data

score = 0.0
bot = auth.auth(log_mod=2)  # no log

FO_CNT_FILE_PATTERN = '/tmp/%s_fo'
FOER_CNT_FILE_PATTERN = '/tmp/%s_foer'


def update_score(score, fo_cnt, foer_cnt):
    score += 1.0 * foer_cnt / (1 + fo_cnt)
    return score


def get_cached_fo_and_foer_cnt(name):
    fo_cnt_filename = FO_CNT_FILE_PATTERN % name
    foer_cnt_filename = FOER_CNT_FILE_PATTERN % name
    with open(fo_cnt_filename, 'r') as f:
        fo_cnt = int(f.read().strip())
    with open(foer_cnt_filename, 'r') as f:
        foer_cnt = int(f.read().strip())
    return fo_cnt, foer_cnt


def get_fo_and_foer_cnt(u):
    try:
        return get_cached_fo_and_foer_cnt(u)
    except:
        pass
    foer_cnt, fo_cnt = user_utils.get_follow_counts(u)
    return fo_cnt, foer_cnt  # note: order is reversed


if len(sys.argv) > 1:
    u = sys.argv[1]
    uid = user_utils.get_user_id(u)
else:
    u = secret_reader.load_secrets()[0]
    uid = user_utils.get_user_id(u)

for fid, fname in user_utils.get_all_followers_gen(bot, uid):
    data.set_followed_back(u, fid)
    fo_cnt, foer_cnt = get_fo_and_foer_cnt(fname)
    if fo_cnt != None and foer_cnt != None:
        score = update_score(score, fo_cnt, foer_cnt)
    # if we run this for every user, and every user has thousands of followers, we will hit rate limit soon and
    # it does not scale. We relax the algo to calculate score once every day, and put gap between data fetching.
    # When followers number further grows, we'll need to relax the frequency even more, e.g. calculate once every 2 days
    time.sleep(1)  # data fetch has a 1s delay, we make it 2s

print('score:', score)
