import requests
import logging
from BeautifulSoup import BeautifulSoup, SoupStrainer
from Queue import Queue
from datetime import datetime


logger = logging.getLogger('crawl_jianshu')
hdlr = logging.FileHandler('./' + str(datetime.now()))
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


def crawl(start_url):
    visited = set()
    q = Queue()
    q.put(start_url)

    tried = 0
    while not q.empty():
        url = q.get()
        if '#' in url:
            url = url[:url.find('#')]
        if '?' in url:
            url = url[:url.find('?')]

        if url in visited:
            logger.info('already visited %s' % url)
            continue

        print 'take %d, crawl %s, remain %d in q' % (tried, url, q.qsize())
        logger.info('crawl %s' % url)
        visited.add(url)
        tried += 1

        try:
            text = requests.get(url).text
        except Exception:
            print 'error connecting'
            logger.info('error connecting')
            continue

        for next_url in get_links(text):
            logger.info('found %s', next_url)
            if next_url.startswith('/'):
                next_url = 'http://jianshu.io' + next_url
                logger.info('adding %s to q' % next_url)
                q.put(next_url)

        for h in logger.handlers:
            h.flush()



start_url = 'http://jianshu.io'
crawl(start_url)
