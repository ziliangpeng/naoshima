import sys
import time

from runners import GenFo, DoFo, GenUnfo, DoUnfo, StealFoers

if __name__ == '__main__':
    # Always DoFo
    t = DoFo()
    t.daemon = True
    t.start()

    # Always DoUnfo
    t = DoUnfo()
    t.daemon = True
    t.start()

    for cmd in sys.argv[1:]:
        if cmd == 'fo':
            t = GenFo()
            t.daemon = True
            t.start()
        elif cmd == 'unfo':
            t = GenUnfo()
            t.daemon = True
            t.start()
        elif cmd.startswith('steal'):
            steal_id = cmd[cmd.index('(') + 1: cmd.index(')')]
            t = StealFoers(steal_id)
            t.daemon = True
            t.start()
        else:
            print 'unrecognized runner', cmd

    while True:
        time.sleep(1)
