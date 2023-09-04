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


class Crawler:
    def __init__(self, seed, interest_prefix=None):
        self.seed = seed
        if interest_prefix is None:
            interest_prefix = [seed]
        self.interest_prefix = interest_prefix
        self.external_links = set()

        self.queue = [seed]
        self.archive = {}
        self.external_links = set()

    def _is_interesting(self, url):
        for prefix in self.interest_prefix:
            if url.startswith(prefix):
                return True
        return False

    def _sanitize_url(self, url):
        return url.split("?")[0]

    def crawl(self):
        while self.queue:
            url = self.queue.pop()
            self.crawl_one_and_expand(url)

        for el in sorted(self.external_links):
            print(el)

    def crawl_website(self, url):
        # Send a GET request to the URL
        response = requests.get(url)

        mime_type = response.headers.get("content-type")
        if not mime_type.startswith("text/html"):
            return "MIME type: {}".format(mime_type), []
        else:
            print("MIME type: {}".format(mime_type))

        # Get the response code
        response_code = response.status_code
        if response_code != 200:
            return "RESPONSE CODE IS {} for {}".format(response_code, url), []

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract all text from the HTML content
        text = soup.get_text()
        links = [link.get("href") for link in soup.find_all("a")]

        return text, links

    def crawl_one_and_expand(self, url):
        queue = self.queue
        archive = self.archive
        external_links = self.external_links

        if url in archive:
            return
        text, links = self.crawl_website(url)
        archive[url] = text

        print("Archive size:", len(archive))
        print(url)
        print(text)
        print("=" * 80)

        for l in links:
            # base_url = SEED
            relative_path = l
            full_url = urljoin(url, relative_path)
            full_url = self._sanitize_url(full_url)
            if not self._is_interesting(full_url):
                external_links.add(full_url)
                continue

            if full_url in archive:
                continue

            self.queue.append(full_url)


NURO_SEED = "https://nuro.ai"
AURORA_SEED = "https://aurora.tech/"

NURO_PREFIX = [
    "https://medium.com/nuro",
    "https://nuro.ai",
    "https://www.nuro.ai",
    "http://nuro.ai",
    "http://www.nuro.ai",
]


def main():
    crawler = Crawler(AURORA_SEED)
    # crawler = Crawler(NURO_SEED, NURO_PREFIX)
    crawler.crawl()


if __name__ == "__main__":
    main()
