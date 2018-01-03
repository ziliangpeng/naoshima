import requests
import sys
from collections import Counter


def _related_tags(target_tag):
    url = 'https://www.instagram.com/explore/tags/%s/?__a=1' % (target_tag)
    r = requests.get(url)
    j = r.json()
    captions = [n['node']['edge_media_to_caption']['edges'][0]['node']['text'] for n in j['graphql']['hashtag']['edge_hashtag_to_media']['edges']]
    tags = filter(lambda x: x.startswith('#'), '\n'.join(captions).split())
    normalized_tags = []
    for tag in tags:
        subtags = tag.split('#')
        for subtag in subtags:
            subtag = subtag.strip()
            if subtag != '' and subtag != target_tag:
                normalized_tags.append('#' + subtag)
    counted_tags = list(Counter(normalized_tags).items())
    counted_tags.sort(key=lambda x: x[1], reverse=True)
    return counted_tags


def all_related(tag):
    return [x[0] for x in _related_tags(tag)]

def top_related(tag, k):
    return [x[0] for x in _related_tags(tag)[:k]]


if __name__ == '__main__':
    tag = sys.argv[1]
    print(top_related(tag, 15))
