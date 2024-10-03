import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse

def crawl_wikipedia(start_url, output_directory, max_pages=100, delay=1):
    visited_urls = set()
    queue = [start_url]
    page_count = 0

    while queue and page_count < max_pages:
        url = queue.pop(0)
        if url in visited_urls:
            continue

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title and content
            title = soup.find('h1', {'id': 'firstHeading'}).text
            content = soup.find('div', {'id': 'mw-content-text'}).text

            # Save content to file
            filename = f"{title.replace(' ', '_')}.txt"
            filepath = os.path.join(output_directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n\nURL: {url}\n\nContent:\n{content}")

            print(f"Saved: {filename}")

            # Extract links to other Wikipedia pages
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin('https://en.wikipedia.org', href)
                    if full_url not in visited_urls:
                        queue.append(full_url)

            visited_urls.add(url)
            page_count += 1

            # Respect Wikipedia's robots.txt with a delay
            time.sleep(delay)

        except Exception as e:
            print(f"Error crawling {url}: {e}")

    print(f"Crawling completed. Processed {page_count} pages.")

if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    output_directory = "wikipedia_content"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    crawl_wikipedia(start_url, output_directory, max_pages=1000000, delay=0.000001)
