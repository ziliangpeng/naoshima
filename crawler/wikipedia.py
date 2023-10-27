import os
import random
import time
import requests
from bs4 import BeautifulSoup
import click
from collections import defaultdict

from loguru import logger

import urllib.parse

WIKI_PAGES_DIR = 'wiki-pages'

START_URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"
# 阿姆斯特丹
START_URL = "https://zh.wikipedia.org/wiki/%E9%98%BF%E5%A7%86%E6%96%AF%E7%89%B9%E4%B8%B9"
# 人工神经网络
START_URL = "https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C"


def decode_url(url):
    decoded_bytes = urllib.parse.unquote_to_bytes(url)
    decoded_text = decoded_bytes.decode('utf-8', errors='replace')
    return decoded_text


def get_wiki_page(url):
    logger.info(f"Crawling {decode_url(url)}")
    response = requests.get(url)
    page_content = response.content
    soup = BeautifulSoup(page_content, 'html.parser')
    title = soup.find('h1', {'class': 'firstHeading'}).text
    text = f'Title: {title}\n'
    paragraphs = soup.find_all('p')
    body = "".join([p.text for p in paragraphs])
    text += f'Context:{body}'

    title = title.replace('/', '-')
    with open(f'{WIKI_PAGES_DIR}/{title}.txt', 'w') as f:
        f.write(text)

    host = url[:url.find('wikipedia.org/')+len('wikipedia.org/')-1]


    # Find links to other Wikipedia pages
    links = [a['href'] for a in soup.find_all('a', href=True) if 'wiki' in a['href'] and ':' not in a['href']]
    # TODO: a flag to control whether to use set or list.
    pages = set([])
    for link in links:
        if not link.startswith('/wiki/'):
            continue
        next_page = f'{host}{link}'
        # print(f'Next: {next_page}')
        pages.add(next_page)
        # time.sleep(1)  # Sleep for 1 second to be respectful to Wikipedia's servers
        # crawl_wikipedia(next_page, depth-1)

    return pages

def find_top_k(queue, k):
    ret =  sorted(queue, key=lambda x: queue[x], reverse=True)[:k]
    for k in ret:
        item = k[k.find('wiki/')+len('wiki/'):]
        item = decode_url(item)
        logger.info(f"{item}: {queue[k]}")
    # logger.info(ret)
    return ret


@click.command()
@click.option('--count', default=10, help='')
def main(count):
    if not os.path.exists(WIKI_PAGES_DIR):
        os.makedirs(WIKI_PAGES_DIR)
    queue = defaultdict(int)
    queue[START_URL] = 1
    done = set()
    num_to_crawl = count
    for _ in range(num_to_crawl):
        logger.info(f'{_} ===============================')
        topk = find_top_k(queue, 5)
        url = random.choice(topk)
        # url = queue.pop()
        queue.pop(url)
        done.add(url)
        links = get_wiki_page(url)
        for link in links:
            if link not in done:
                queue[link] += 1
        time.sleep(1)  # Sleep for 1 second to be respectful to Wikipedia's servers

if __name__ == '__main__':
    main()