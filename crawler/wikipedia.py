import os
import random
import time
import requests
from bs4 import BeautifulSoup
import click
from collections import defaultdict

from loguru import logger

import urllib.parse

import pickle

WIKI_PAGES_DIR = 'wiki-pages'

START_URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"
# 阿姆斯特丹
# START_URL = "https://zh.wikipedia.org/wiki/%E9%98%BF%E5%A7%86%E6%96%AF%E7%89%B9%E4%B8%B9"
# 人工神经网络
START_URL = "https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C"


def decode_url(url):
    decoded_bytes = urllib.parse.unquote_to_bytes(url)
    decoded_text = decoded_bytes.decode('utf-8', errors='replace')
    return decoded_text


def lang(url):
    return url[url.find('://')+3:url.find('wikipedia.org/')-1]

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
    target_dir = WIKI_PAGES_DIR + '/' + lang(url)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(f'{target_dir}/{title}.txt', 'w') as f:
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
@click.option('--load', default=False, help='')
@click.option('--k', default=5, help='')
def main(count, load, k):
    # Load queue and done from file if they exist
    if load and os.path.exists('queue.pkl'):
        logger.info("loading queue.pkl")
        with open('queue.pkl', 'rb') as f:
            queue = pickle.load(f)
    else:
        queue = defaultdict(int)
        queue[START_URL] = 1

    if load and os.path.exists('done.pkl'):

        with open('done.pkl', 'rb') as f:
            done = pickle.load(f)
    else:
        done = set()

    num_to_crawl = count
    for _ in range(num_to_crawl):
        logger.info(f'{_} ===============================')
        topk = find_top_k(queue, k)
        url = random.choice(topk)
        # url = queue.pop()
        queue.pop(url)
        done.add(url)
        links = get_wiki_page(url)
        for link in links:
            if link not in done:
                queue[link] += 1

        # Save queue and done to file
        with open('queue.pkl', 'wb') as f:
            pickle.dump(queue, f)

        with open('done.pkl', 'wb') as f:
            pickle.dump(done, f)

        logger.info("persisted queue.pkl and done.pkl. Sleeping for 1 second")

        time.sleep(1)  # Sleep for 1 second to be respectful to Wikipedia's servers

if __name__ == '__main__':
    main()