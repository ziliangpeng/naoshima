import datetime
import logs
import time
from datetime import timedelta

import user_utils
import lang_utils

date = datetime.datetime.now

logger = logs.logger


class Filter:
    def __init__(self, u, conditions={}):
        self.u = str(u)
        self.conditions = conditions
        self.methods = {
            'max_follows': self.max_follows,
            'min_ratio': self.min_ratio,
            'max_ratio': self.max_ratio,
            'has_lang': self.has_lang,
            'fresh': self.fresh
        }

    def apply(self):
        for k, v in list(self.conditions.items()):
            if not self.methods[k](v):
                return False
        return True

    def has_lang(self, lang):
        langs = lang_utils.langs(self.u)
        speaks_lang = lang in [l.lang for l in langs]
        if not speaks_lang:
            logger.debug('User %s does not speak %s', self.u, lang)
            return False
        return True

    def max_follows(self, threshold):
        followed_by_count, follows_count = \
            user_utils.get_follow_counts(self.u)
        if followed_by_count is None or follows_count is None:
            logger.info('error occurred when investigating %s', self.u)
            return False
        if follows_count > threshold:
            logger.debug('follower %s has %d follows(>%d). It is an overwhelmed stalker',
                         self.u, follows_count, threshold)
            return False
        return True

    def min_ratio(self, ratio_threshold):
        followed_by_count, follows_count = \
            user_utils.get_follow_counts(self.u)
        if followed_by_count is None or follows_count is None:
            logger.error('error occurred when investigating %s', self.u)
            return False
        if follows_count < followed_by_count * ratio_threshold:
            logger.debug('follower %s has %d follows and %d followed_by(<%f). Not likely to follow back',
                         self.u, follows_count, followed_by_count, ratio_threshold)
            return False
        return True

    def max_ratio(self, ratio_threshold):
        followed_by_count, follows_count = \
            user_utils.get_follow_counts(self.u)
        if followed_by_count is None or follows_count is None:
            print('error occurred when investigating ', self.u)
            return False
        if follows_count > followed_by_count * ratio_threshold:
            logger.debug('follower %s has %d follows and %d followed_by(>%f). Over-following.',
                         self.u, follows_count, followed_by_count, ratio_threshold)
            return False
        return True

    def fresh(self, fresh_threshold):
        recent_post_epoch = user_utils.get_recent_post_epoch(self.u)
        now_epoch = int(time.time())
        epoch_diff = timedelta(seconds=now_epoch - recent_post_epoch)
        td_threshold = timedelta(seconds=fresh_threshold)
        if epoch_diff > td_threshold:
            logger.debug('follower %s posted %s ago. longer than %s',
                         self.u, epoch_diff, td_threshold)
            return False
        return True
