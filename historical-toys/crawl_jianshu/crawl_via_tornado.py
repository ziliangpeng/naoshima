import requests
import logging
from BeautifulSoup import BeautifulSoup, SoupStrainer
from Queue import Queue
from datetime import datetime
from tornado.httpclient import AsyncHTTPClient
from tornado import ioloop
from threading import Semaphore, Lock


logger = logging.getLogger('crawl_jianshu')
hdlr = logging.FileHandler('./tornado_' + str(datetime.now()))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


def get_links(text):
    for link in BeautifulSoup(text, parseOnlyThese=SoupStrainer('a')):
        try:
            yield link['href']
        except KeyError:
            pass


visited = set()
#q = Queue()
#semaphore = Semaphore(5000)
#print_lock = Lock()
#set_lock = Lock()
tried = 0
done = 0

def handle_request(response):
    print 'enter handle request'
    #semaphore.release()
    global tried
    global done

    done += 1
    if response.error:
        print response.error
        pass
    else:
        text = response.body
        for url in get_links(text):
            if url.startswith('/'):
                url = 'http://jianshu.io' + url
                if '#' in url:
                    url = url[:url.find('#')]
                if '?' in url:
                    url = url[:url.find('?')]

                if url not in visited:
                    visited.add(url)
                    print 'take %d, crawl %s, done %d' % (tried, url, done)
                    logger.info('crawl %s' % url)
                    http_client.fetch(url, handle_request)
                    tried += 1


start_url = 'http://jianshu.io'
http_client = AsyncHTTPClient()
#semaphore.acquire()
print 'fetch'
http_client.fetch(start_url, handle_request)

print 'start'
ioloop.IOLoop.instance().start()

