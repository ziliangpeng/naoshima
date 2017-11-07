import utils
import auth
import requests
import random

score = 0.0
bot = auth.auth(log_mod=2)  # no log

FO_CNT_FILE_PATTERN = '/tmp/%s_fo'
FOER_CNT_FILE_PATTERN = '/tmp/%s_foer'
INVALIDATE_CACHE_RATE = 0.01


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


def write_cached_fo_and_foer_cnt(name, fo_cnt, foer_cnt):
    fo_cnt_filename = FO_CNT_FILE_PATTERN % name
    foer_cnt_filename = FOER_CNT_FILE_PATTERN % name
    with open(fo_cnt_filename, 'w+') as f:
        f.write(str(fo_cnt))
    with open(foer_cnt_filename, 'w+') as f:
        f.write(str(foer_cnt))
    return fo_cnt, foer_cnt


def get_fo_and_foer_cnt(f):
    if random.random() > INVALIDATE_CACHE_RATE:
        try:
            return get_cached_fo_and_foer_cnt(f)
        except:
            pass

    url = 'https://www.instagram.com/%s/?__a=1' % f
    r = requests.get(url)
    if r.status_code == 200:
        j = r.json()
        fo_cnt = j["user"]["follows"]["count"]
        foer_cnt = j["user"]["followed_by"]["count"]
        write_cached_fo_and_foer_cnt(f, fo_cnt, foer_cnt)
        return fo_cnt, foer_cnt
    else:
        return -1, -1


for fid, fname in utils.get_all_followers_gen(bot):
    fo_cnt, foer_cnt = get_fo_and_foer_cnt(fname)
    if fo_cnt != -1 and foer_cnt != -1:
        score = update_score(score, fo_cnt, foer_cnt)

print 'score:', score
