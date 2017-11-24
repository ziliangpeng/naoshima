import sys
import time
import secret_reader
import data_repo

from runners import DoFo, GenUnfo, DoUnfo, StealFoers, Fofo, StealSuperBrand, StealSimilarTo


d = data_repo.datas[0]
u = d.u

if __name__ == '__main__':
    # Always DoFo
    t = DoFo(u)
    t.daemon = True
    t.start()

    # Always DoUnfo
    t = DoUnfo(u)
    t.daemon = True
    t.start()

    cmds = secret_reader.load_commands()

    for cmd in cmds:
        if cmd == 'unfo':
            t = GenUnfo(u)
            t.daemon = True
            t.start()
        elif cmd.startswith('steal('):
            steal_name = cmd[cmd.index('(') + 1: cmd.index(')')]
            t = StealFoers(u, steal_name)
            t.daemon = True
            t.start()
        elif cmd.startswith('stealsimilar('):
            steal_name = cmd[cmd.index('(') + 1: cmd.index(')')]
            t = StealSimilarTo(u, steal_name)
            t.daemon = True
            t.start()
        elif cmd == 'fofo':
            t = Fofo(u)
            t.daemon = True
            t.start()
        elif cmd == 'superbrand' or cmd == 'sb':
            t = StealSuperBrand(u)
            t.daemon = True
            t.start()
        else:
            print('unrecognized runner', cmd)

    while True:
        time.sleep(1)
