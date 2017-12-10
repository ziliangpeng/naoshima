import auth
import config_reader
import user_utils
from queue import Queue


class Data:
    pass


FO_QUEUE_SIZE = 50
UNFO_QUEUE_SIZE = 50


datas = {}
d1 = Data()
d1.queue_to_fo = Queue(FO_QUEUE_SIZE)
d1.queue_to_unfo = Queue(UNFO_QUEUE_SIZE)
d1.bot = auth.auth()  # should take creds
d1.u = config_reader.load_secrets()[0]
d1.user_id = user_utils.get_user_id(d1.u)  # TODO: check null
d1.conditions = config_reader.load_conditions()
d1.like_per_fo = config_reader.load_like_per_fo()
d1.comment_pool = config_reader.load_comment_pool()
datas[d1.u] = d1
datas[0] = d1
