import requests

import auth
import secret_reader
# import utils
import user_utils

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


uid = secret_reader.load_user_id()
for fid, fname in user_utils.get_all_followers_gen(bot, uid):
    fo_cnt, foer_cnt = get_fo_and_foer_cnt(fname)
    if fo_cnt != None and foer_cnt != None:
        score = update_score(score, fo_cnt, foer_cnt)

print('score:', score)
