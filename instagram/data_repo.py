import auth
from model import UniqueQueue


FO_QUEUE_SIZE = 50
UNFO_QUEUE_SIZE = 50

queue_to_fo = UniqueQueue(FO_QUEUE_SIZE)
queue_to_unfo = UniqueQueue(UNFO_QUEUE_SIZE)
bot = auth.auth()

id_name_dict = {}
poked = set()  # ppl I've followed before
