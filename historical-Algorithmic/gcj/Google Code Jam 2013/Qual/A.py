from Queue import Queue, Empty
from threading import Thread
from time import sleep

def read_array(convertor=None):
    ret = raw_input().split()
    if convertor: ret = map(convertor, ret)
    return ret

work_queue = Queue()
result_queue = []

#======================== SOF  =============================#

class Work:
    def __init__(self, id):
        self.id = id

    def solve(self):
        result = self._run()
        result_queue.append((self.id, result))

class Solver(Work):
    def __init__(self, id, data_container):
        Work.__init__(self, id)
        self.data_container = data_container

    def win(self, status, player):
        for c in status:
            if c not in [player, 'T']:
                return False
        return True

    def _run(self):
        game = self.data_container.game

        players = {'X': False, 'O', False}
        X_win = O_win = False
        # horizontal
        for i in range(4):
            status = game[i]
            for player in players.keys():
                if self.win(status, player):
                    players[player] = True

        # vertical
        for i in range(4):
            status = ''.join([row[i] for row in game])
            for player in players.keys():
                if self.win(status, player):
                    players[player] = True

        # 
        status = ''.join([game[i][i] for i in range(4)])
        for player in players.keys():
            if self.win(status, player):
                players[player] = True

        status = ''.join([game[i][3-i] for i in range(4)])
        for player in players.keys():
            if self.win(status, player):
                players[player] = True

        if players['X']:
            return 'X won'
        elif players['O']:
            return 'O won'
        elif '.' in ''.join(game):
            return 'Game has not completed'
        else:
            return 'Draw'


class DataContainer:
    def __init__(self, id):
        if id:
            raw_input()
        self.game = [raw_input() for i in range(4)]
        

#======================== EOF  =============================#

NUM_THREAD = 3
class Executor(Thread):
    def run(self):
        while True:
            try:
                work = work_queue.get_nowait()
                work.solve()
                work_queue.task_done()
            except Empty:
                return

def main():
    T = input()
    executors = [Executor() for i in range(NUM_THREAD)]
    for t in range(T):
        data_container = DataContainer(t)
        work_queue.put(Solver(t, data_container))
    for executor in executors:
        executor.start()

    for executor in executors:
        executor.join()
    global result_queue
    result_queue = sorted(result_queue)
    for id, result in result_queue:
        print 'Case #%d:' % (id+1),
        print str(result)


if __name__ == '__main__':
    main()



