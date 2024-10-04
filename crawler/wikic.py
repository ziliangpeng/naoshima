import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse
import random
from collections import Counter

def crawl_wikipedia(start_url, output_directory, max_pages=100, delay=1, k=5):
    visited_urls = set()
    url_counter = Counter()
    queue = [(start_url, 0)]
    page_count = 0

    # Extract the language code from the start_url
    parsed_url = urlparse(start_url)
    lang_code = parsed_url.netloc.split('.')[0]

    while queue and page_count < max_pages:
        # Sort queue by count and select randomly from top k
        queue.sort(key=lambda x: x[1], reverse=True)
        top_k = min(k, len(queue))
        url, _ = random.choice(queue[:top_k])
        queue = [(u, c) for u, c in queue if u != url]

        if url in visited_urls:
            continue

        from urllib.parse import unquote
        decoded_url = unquote(url)
        print(f"Processing URL: {decoded_url} (Count: {url_counter[url]})")

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title and content
            title = soup.find('h1', {'id': 'firstHeading'}).text
            content = soup.find('div', {'id': 'mw-content-text'}).text

            # Save content to file
            filename = f"{title.replace(' ', '_')}.txt"
            lang_directory = os.path.join(output_directory, lang_code)
            if not os.path.exists(lang_directory):
                os.makedirs(lang_directory)
            filename = filename.replace('/', '_')
            filepath = os.path.join(lang_directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n\nURL: {url}\n\nContent:\n{content}")

            # print(f"Saved: {filename}")

            # Extract links to other Wikipedia pages
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin(f'https://{lang_code}.wikipedia.org', href)
                    if full_url not in visited_urls:
                        url_counter[full_url] += 1
                        queue.append((full_url, url_counter[full_url]))

            visited_urls.add(url)
            page_count += 1

            # Respect Wikipedia's robots.txt with a delay
            time.sleep(delay)

        except Exception as e:
            print(f"Error crawling {url}: {e}")

    print(f"Crawling completed. Processed {page_count} pages.")

if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    start_url = "https://zh.wikipedia.org/wiki/%E6%84%8F%E5%A4%A7%E5%88%A9"
    output_directory = "wikipedia_content"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    crawl_wikipedia(start_url, output_directory, max_pages=1000000, delay=0.1, k=5)
