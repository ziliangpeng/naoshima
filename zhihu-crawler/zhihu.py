import requests
import re
import time


class UniqueQueue:

    def __init__(self):
        import Queue
        self.q = Queue.Queue()
        self._items = set()

    def put(self, item):
        if item not in self._items:
            self._items.add(item)
            self.q.put(item)

    def get(self):
        return self.q.get()

    def size(self):
        return self.q.qsize()

    def empty(self):
        return self.q.empty()


def analyse(url, visited=set()):
    if url in visited:
        print 'visited', url
        return set(), set()
    visited.add(url)
    print 'visiting', url

    cookies = load_cookies()
    """ 
        user pattern: 
            <a href="/people/ziliang">
            href="http://www.zhihu.com/people/mai-liao

        question pattern: 
            href="/question/23830784#answer-6073033"
    """
    USER_PATTERN = '(?<=href=\"/people/)[a-zA-Z0-0_\-]+(?=\">)'
    Q_PATTERN = '(?<=href=\"/question/)\d+'
    response = requests.get(url, cookies=cookies)
    users = set(re.findall(USER_PATTERN, response.text))
    questions = set(re.findall(Q_PATTERN, response.text))

    return users, questions


def update_users(users, new_users):
    USER_FILE = 'users.txt'
    f = open(USER_FILE, 'a')
    for user in new_users:
        if user not in users:
            users.add(user)
            f.write(user + '\n')
    f.close()

def crawl_questions():
    q = UniqueQueue()
    start_url = 'http://www.zhihu.com'
    q.put(start_url)
    all_users = set()
    
    while not q.empty():
        url = q.get()
        users, questions = analyse(url)
        update_users(all_users, users)
        for question in questions:
            q.put('http://www.zhihu.com/question/%s' % question)
        print 'user size', len(all_users), 'question size', q.size()

        time.sleep(10)


main()
