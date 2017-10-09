import sys
import time
import signal
import auth
import secret_reader
from model import UniqueQueue
from runners import GenFo, DoFo, GenUnfo, DoUnfo, StealFoers


WHITELIST_USER = secret_reader.load_whitelist()
FO_QUEUE_SIZE = 50
UNFO_QUEUE_SIZE = 50
USER_ID = secret_reader.load_user_id()

bot = auth.auth()
queue_to_fo = UniqueQueue(FO_QUEUE_SIZE)
queue_to_unfo = UniqueQueue(UNFO_QUEUE_SIZE)
id_name_dict = {}
poked = set()  # ppl I've followed before


def signal_handler(signal, frame):
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    # Always DoFo
    t = DoFo(bot, queue_to_fo, id_name_dict)
    t.daemon = True
    t.start()

    # Always DoUnfo
    t = DoUnfo(bot, queue_to_unfo)
    t.daemon = True
    t.start()

    for cmd in sys.argv[1:]:
        if cmd == 'fo':
            t = GenFo(bot, queue_to_fo, id_name_dict, poked)
            t.daemon = True
            t.start()
        elif cmd == 'unfo':
            t = GenUnfo(bot, queue_to_unfo, id_name_dict)
            t.daemon = True
            t.start()
        elif cmd.startswith('steal'):
            steal_id = cmd[cmd.index('(') + 1: cmd.index(')')]
            t = StealFoers(bot, steal_id, queue_to_fo, id_name_dict)
            t.daemon = True
            t.start()

    while True:
        time.sleep(1)
