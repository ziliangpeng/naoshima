import auth
import secret_reader
import utils
from model import UniqueQueue


class Data:
    pass


FO_QUEUE_SIZE = 50
UNFO_QUEUE_SIZE = 50

queue_to_fo = UniqueQueue(FO_QUEUE_SIZE)
queue_to_unfo = UniqueQueue(UNFO_QUEUE_SIZE)
bot = auth.auth()

id_name_dict = {}
poked = set()  # ppl I've followed before



datas = {}
d1 = Data()
d1.queue_to_fo = UniqueQueue(FO_QUEUE_SIZE)
d1.queue_to_unfo = UniqueQueue(UNFO_QUEUE_SIZE)
d1.bot = auth.auth()  # should take creds
d1.poked = set()
d1.username = secret_reader.load_secrets()[0]
d1.user_id = utils.get_user_id(d1.username)
datas[d1.username] = d1
