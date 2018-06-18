import auth
import user_config_reader
# import user_utils
from queue import Queue


class Data:
    pass


FO_QUEUE_SIZE = 50
UNFO_QUEUE_SIZE = 50


d0 = Data()
d0.queue_to_fo = Queue(FO_QUEUE_SIZE)
d0.queue_to_unfo = Queue(UNFO_QUEUE_SIZE)
d0.bot = auth.auth()  # should take creds
d0.u = user_config_reader.load_secrets()[0]
# d0.user_id = user_utils.get_user_id(d0.u)  # TODO: check null
d0.conditions = user_config_reader.load_conditions()
d0.like_per_fo = user_config_reader.load_like_per_fo()
d0.comment_pool = user_config_reader.load_comment_pool()


if __name__ == '__main__':
    d = d0
    print('u', d.u)
    print('conditions', d.conditions)
    print('like_per_fo', d.like_per_fo)
