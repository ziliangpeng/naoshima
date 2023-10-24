import os
import time
import requests
from bs4 import BeautifulSoup
import click

from loguru import logger

WIKI_PAGES_DIR = 'wiki-pages'

START_URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"

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


    # Find links to other Wikipedia pages
    links = [a['href'] for a in soup.find_all('a', href=True) if 'wiki' in a['href'] and ':' not in a['href']]
    pages = []
    for link in links:
        if not link.startswith('/wiki/'):
            continue
        next_page = f'https://en.wikipedia.org{link}'
        # print(f'Next: {next_page}')
        pages.append(next_page)
        # time.sleep(1)  # Sleep for 1 second to be respectful to Wikipedia's servers
        # crawl_wikipedia(next_page, depth-1)

    return pages


@click.command()
@click.option('--count', default=10, help='')
def main(count):
    if not os.path.exists(WIKI_PAGES_DIR):
        os.makedirs(WIKI_PAGES_DIR)
    queue = set([START_URL])
    done = set()
    num_to_crawl = count
    for _ in range(num_to_crawl):
        url = queue.pop()
        done.add(url)
        links = get_wiki_page(url)
        for link in links:
            if link not in done and link not in queue:
                queue.add(link)
        time.sleep(1)  # Sleep for 1 second to be respectful to Wikipedia's servers

if __name__ == '__main__':
    main()