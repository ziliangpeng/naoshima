import datetime
import time
from datetime import timedelta

import user_utils

date = datetime.datetime.now


class Filter:
    def __init__(self, name, conditions={}):
        self.name = str(name)
        self.conditions = conditions
        self.methods = {
            'max_follows': self.max_follows,
            'min_ratio': self.min_ratio,
            'fresh': self.fresh
        }

    def apply(self):
        for k, v in list(self.conditions.items()):
            if not self.methods[k](v):
                return False
        return True

    def max_follows(self, threshold):
        followed_by_count, follows_count = \
            user_utils.get_follow_counts(self.name)
        if followed_by_count == None or follows_count == None:
            print('error occurred when investigating ', self.name)
            return False
        if follows_count > threshold:
            print('%s: follower %s has %d follows(>%d). It is an overwhelmed stalker' % \
                  (date(), self.name, follows_count, threshold))
            return False
        return True

    def min_ratio(self, ratio_threshold):
        followed_by_count, follows_count = \
            user_utils.get_follow_counts(self.name)
        if followed_by_count == None or follows_count == None:
            print('error occurred when investigating ', self.name)
            return False
        if follows_count < followed_by_count * ratio_threshold:
            print('%s: follower %s has %d follows and %d followed_by(<%f). Not likely to follow back' % \
                  (date(), self.name, follows_count, followed_by_count, ratio_threshold))
            return False
        return True

    def fresh(self, fresh_threshold):
        recent_post_epoch = user_utils.get_recent_post_epoch(self.name)
        now_epoch = int(time.time())
        epoch_diff = timedelta(seconds=now_epoch - recent_post_epoch)
        td_threshold = timedelta(seconds=fresh_threshold)
        if epoch_diff > td_threshold:
            print('%s: follower %s posted %s ago. longer than %s' % \
                  (date(), self.name, epoch_diff, td_threshold))
            return False
        return True
