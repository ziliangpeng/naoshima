import sys
import time
import secret_reader

from runners import GenFo, DoFo, GenUnfo, DoUnfo, StealFoers, Fofo, StealSuperBrand


username = secret_reader.load_secrets()[0]

if __name__ == '__main__':
    # Always DoFo
    t = DoFo(username)
    t.daemon = True
    t.start()

    # Always DoUnfo
    t = DoUnfo(username)
    t.daemon = True
    t.start()

    for cmd in sys.argv[1:]:
        if cmd == 'unfo':
            t = GenUnfo(username)
            t.daemon = True
            t.start()
        elif cmd.startswith('steal'):
            steal_name = cmd[cmd.index('(') + 1: cmd.index(')')]
            t = StealFoers(username, steal_name)
            t.daemon = True
            t.start()
        elif cmd == 'fofo':
            t = Fofo(username)
            t.daemon = True
            t.start()
        elif cmd == 'superbrand' or cmd == 'sb':
            t = StealSuperBrand(username)
            t.daemon = True
            t.start()
        else:
            print('unrecognized runner', cmd)

    while True:
        time.sleep(1)
