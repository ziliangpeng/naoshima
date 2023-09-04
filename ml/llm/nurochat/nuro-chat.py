"""
1. Crawl all Nuro.ai data
2. Use LlamaIndex to index text and ask questions.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


"""
http://nuro.ai/firstresponders
MIME type: application/pdf

Others:
text/html; charset=utf-8
"""

def crawl_website(url):
    # Send a GET request to the URL
    response = requests.get(url)

    mime_type = response.headers.get('content-type')
    if not mime_type.startswith('text/html'):
        return "MIME type: {}".format(mime_type), []
    else:
        print("MIME type: {}".format(mime_type))
    
    # Get the response code
    response_code = response.status_code
    if response_code != 200:
        return "RESPONSE CODE IS {} for {}".format(response_code, url), []
    

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all text from the HTML content
    text = soup.get_text()
    links = [link.get('href') for link in soup.find_all('a')]

    return text, links

INTEREST_PREFIX = ['https://medium.com/nuro', 'https://nuro.ai', 'https://www.nuro.ai', 'http://nuro.ai', 'http://www.nuro.ai']

def is_interesting(url):
    for prefix in INTEREST_PREFIX:
        if url.startswith(prefix):
            return True
    return False

def sanitize_url(url):
    return url.split('?')[0]

SEED = 'https://nuro.ai'

def crawl_one_and_expand(url, queue, archive, external_links):
    if url in archive:
        return
    text, links = crawl_website(url)
    archive[url] = text

    print("Archive size:", len(archive))
    print(url)
    print(text)
    print('=' * 80)

    for l in links:
        # base_url = SEED
        relative_path = l
        full_url = urljoin(url, relative_path)
        full_url = sanitize_url(full_url)
        if not is_interesting(full_url):
            external_links.add(full_url)
            continue

        if full_url in archive:
            continue

        queue.append(full_url)
        # print(full_url)  # Output: https://example.com/path/to/resource

def crawl_nuro():
    queue = [SEED]
    archive = {}
    external_links = set()
    while queue:
        url = queue.pop()
        crawl_one_and_expand(url, queue, archive, external_links)

    external_links = sorted(external_links)
    for el in external_links:
        print(el)

def main():
    crawl_nuro()

if __name__ == '__main__':
    main()