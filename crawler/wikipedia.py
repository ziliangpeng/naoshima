import os
import random
import time
import requests
from bs4 import BeautifulSoup
import click
from collections import defaultdict

from loguru import logger

WIKI_PAGES_DIR = 'wiki-pages'

START_URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"
# START_URL = "https://zh.wikipedia.org/wiki/%E9%98%BF%E5%A7%86%E6%96%AF%E7%89%B9%E4%B8%B9"

def get_wiki_page(url):
    logger.info(f"Crawling {url}")
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
    pages = []
    for link in links:
        if not link.startswith('/wiki/'):
            continue
        next_page = f'{host}{link}'
        # print(f'Next: {next_page}')
        pages.append(next_page)
        # time.sleep(1)  # Sleep for 1 second to be respectful to Wikipedia's servers
        # crawl_wikipedia(next_page, depth-1)

    return pages

def find_top_k(queue, k):
    ret =  sorted(queue, key=lambda x: queue[x], reverse=True)[:k]
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
        topk = find_top_k(queue, 5)
        url = random.choice(topk)
        # url = queue.pop()
        queue.pop(url)
        done.add(url)
        links = get_wiki_page(url)
        for link in links:
            if link not in done and link not in queue:
                queue[link] += 1
        time.sleep(1)  # Sleep for 1 second to be respectful to Wikipedia's servers

if __name__ == '__main__':
    main()