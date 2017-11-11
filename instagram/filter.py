import datetime
from datetime import timedelta
import time
import utils

date = datetime.datetime.now


class Filter:
    def __init__(self, name, conditions={}):
        self.name = str(name)

    def max_follows(self, threshold):
        followed_by_count, follows_count = \
            utils.get_follow_counts(self.name, (0, 0))
        if follows_count > threshold:
            print '%s: follower %s has %d follows(>%d). It is an overwhelmed stalker' % \
                  (date(), self.name, follows_count, threshold)
            return False
        return True

    def min_ratio(self, ratio_threshold):
        followed_by_count, follows_count = \
            utils.get_follow_counts(self.name, (0, 0))
        if follows_count < followed_by_count * ratio_threshold:
            print '%s: follower %s has %d follows and %d followed_by(<%f). Not likely to follow back' % \
                  (date(), self.name, follows_count, followed_by_count, ratio_threshold)
            return False
        return True

    def fresh(self, fresh_threshold):
        recent_post_epoch = utils.get_recent_post_epoch(self.name, -1)
        now_epoch = int(time.time())
        epoch_diff = timedelta(seconds=now_epoch - recent_post_epoch)
        td_threshold = timedelta(seconds=fresh_threshold)
        if epoch_diff > td_threshold:
            print '%s: follower %s posted %s ago. longer than %s' % \
                  (date(), self.name, epoch_diff, td_threshold)
            return False
        return True


def legacy_filter(name):
    f = Filter(name)

    fresh_threshold = 3600 * 24 * 14  # 14 days
    ratio_threshold = 1.5
    follows_threshold = 5000

    if not f.fresh(fresh_threshold):
        return False
    elif not f.min_ratio(ratio_threshold):
        return False
    elif not f.max_follows(follows_threshold):
        return False
    else:
        return True
