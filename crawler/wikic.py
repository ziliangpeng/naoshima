import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse, unquote, quote
import random
from collections import Counter
from loguru import logger
import pickle
import statsd
from zhconv import convert

# Constant for the state filename
STATE_FILENAME = "crawl_state.pkl"

# Initialize statsd client
statsd_client = statsd.StatsClient()

def sanitize_url(url):
    parsed_url = urlparse(url)
    decoded_url = unquote(parsed_url.geturl())
    converted_url = convert(decoded_url, 'zh-cn')
    return quote(converted_url.split('#')[0])

def save_state(visited_urls, url_counter, queue, page_count, lang_code, output_directory):
    state = {
        'visited_urls': visited_urls,
        'url_counter': url_counter,
        'queue': queue,
        'page_count': page_count,
        'lang_code': lang_code
    }
    state_file = os.path.join(output_directory, STATE_FILENAME)
    with open(state_file, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved state to {state_file}")

def load_state(output_directory):
    state_file = os.path.join(output_directory, STATE_FILENAME)
    if os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        logger.info(f"Loaded state from {state_file}")
        return state['visited_urls'], state['url_counter'], state['queue'], state['page_count'], state['lang_code']
    return set(), Counter(), set(), 0, None

def crawl_wikipedia(start_url, output_directory, max_pages=100, delay=1):
    visited_urls, url_counter, queue, page_count, lang_code = load_state(output_directory)

    visited_urls = set(map(sanitize_url, visited_urls))
    queue = set(map(sanitize_url, queue))
    url_counter = Counter({sanitize_url(k): v for k, v in url_counter.items()})
    logger.info(f"Size of visited_urls: {len(visited_urls)}")
    logger.info(f"Size of url_counter: {len(url_counter)}")
    logger.info(f"Size of queue: {len(queue)}")
    statsd_client.gauge('wikipedia_crawler.page_count', page_count + 1)
    statsd_client.gauge('wikipedia_crawler.queue_size', len(queue))
    statsd_client.gauge('wikipedia_crawler.visited_urls', len(visited_urls))
    
    if not lang_code:
        parsed_url = urlparse(start_url)
        lang_code = parsed_url.netloc.split('.')[0]

    if not queue:
        queue = {start_url}

    last_request_time = 0

    dudup_queue = set()
    for url in queue:
        if url in visited_urls:
            continue
        encoded_url = sanitize_url(url)
        dudup_queue.add(encoded_url)
    logger.info(f"Size of dedup queue: {len(dudup_queue)}")
    queue = dudup_queue

    while queue and page_count < max_pages:
        top_n = min(1024, max(1, int(len(queue) * 0.2)))
        top_urls = sorted(queue, key=lambda x: url_counter[x], reverse=True)[:top_n]
        total_count = sum(url_counter[url] for url in top_urls)
        if total_count > 0:
            probabilities = [url_counter[url] / total_count for url in top_urls]
        else:
            probabilities = [1 / len(top_urls)] * len(top_urls)
        url = random.choices(top_urls, weights=probabilities, k=1)[0]
        queue.remove(url)

        url = url.split('#')[0]

        statsd_client.gauge('wikipedia_crawler.queue_size', len(queue))
        if url in visited_urls:
            logger.info(f"Already visited {url} (Decoded URL: {unquote(url)})")
            continue

        decoded_url = unquote(url)
        simp_decoded_url = convert(decoded_url, 'zh-cn')
        logger.info(f"Page count: {page_count + 1} - Processing URL: {simp_decoded_url} (Count: {url_counter[url]}, Queue size: {len(queue)}, Visited URLs: {len(visited_urls)})")
        if simp_decoded_url.startswith("http://") or simp_decoded_url.startswith("https://"):
            scheme, rest = simp_decoded_url.split("://", 1)
            url = f"{scheme}://{quote(rest)}"
        else:
            url = quote(simp_decoded_url)
        
        # Send metrics to statsd
        statsd_client.gauge('wikipedia_crawler.page_count', page_count + 1)
        statsd_client.gauge('wikipedia_crawler.url_count', url_counter[url])
        statsd_client.gauge('wikipedia_crawler.queue_size', len(queue))
        statsd_client.gauge('wikipedia_crawler.visited_urls', len(visited_urls))


        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.find('h1', {'id': 'firstHeading'}).text
            content = soup.find('div', {'id': 'mw-content-text'}).text

            # Save parsed content
            filename = f"{title.replace(' ', '_')}.txt"
            lang_directory = os.path.join(output_directory, lang_code)
            if not os.path.exists(lang_directory):
                os.makedirs(lang_directory)
            filename = filename.replace('/', '_')
            filepath = os.path.join(lang_directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                epoch_time = int(time.time())
                f.write(f"Title: {title}\n\nURL: {url}\n\nEpoch: {epoch_time}, Current Time: {current_time}\n\nContent:\n{content}")

            # Save raw HTML content
            raw_html_directory = os.path.join(output_directory, f"{lang_code}_full")
            if not os.path.exists(raw_html_directory):
                os.makedirs(raw_html_directory)
            raw_html_filename = f"{title.replace(' ', '_')}.html"
            raw_html_filename = raw_html_filename.replace('/', '_')
            raw_html_filepath = os.path.join(raw_html_directory, raw_html_filename)
            with open(raw_html_filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)

            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin(f'https://{lang_code}.wikipedia.org', href)
                    full_url = sanitize_url(full_url)
                    # if lang_code == 'zh':
                    #     decoded_full_url = unquote(full_url)
                    #     converted_full_url = convert(decoded_full_url, 'zh-cn')
                    #     full_url = quote(converted_full_url)
                    # full_url = full_url.split('#')[0]
                    url_counter[full_url] += 1
                    if full_url not in visited_urls and full_url not in queue:
                        # decoded_full_url = unquote(full_url)
                        # logger.info(f"Adding {decoded_full_url} to queue.")
                        queue.add(full_url)

            visited_urls.add(url)
            queue.discard(url)
            page_count += 1

            if page_count % 100 == 0:
                save_state(visited_urls, url_counter, queue, page_count, lang_code, output_directory)

            current_time = time.time()
            time_since_last_request = current_time - last_request_time
            if time_since_last_request < delay:
                time.sleep(delay - time_since_last_request)
            last_request_time = time.time()

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                visited_urls.add(url)

    logger.info(f"Crawling completed. Processed {page_count} pages.")
    save_state(visited_urls, url_counter, queue, page_count, lang_code, output_directory)
if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    start_url = "https://zh.wikipedia.org/wiki/%E6%84%8F%E5%A4%A7%E5%88%A9"
    output_directory = "wikipedia_content"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    crawl_wikipedia(start_url, output_directory, max_pages=1000000, delay=1)
