import requests
import sys
from collections import Counter
from logs import logger


def _extract_caption(node):
    edges = node['node']['edge_media_to_caption']['edges']
    if len(edges) == 0:
        return ''
    else:
        return edges[0]['node']['text']


def tags_from_caption(caption):
    tags = filter(lambda x: x.startswith('#'), caption.split())
    normalized_tags = []
    for tag in tags:
        subtags = tag.split('#')
        for subtag in subtags:
            subtag = subtag.strip()
            if subtag != '':
                normalized_tags.append('#' + subtag)
    return normalized_tags


def _related_tags(*target_tags):
    target_tags = [t.replace('#', '') for t in target_tags]
    normalized_tags = []
    for target_tag in target_tags:
        url = 'https://www.instagram.com/explore/tags/%s/?__a=1' % (target_tag)
        logger.info("explore tags. url is %s", url)
        r = requests.get(url)
        logger.info("explore tags. response encoding is %s", r.encoding)
        if r.status_code == 200:
            try:
                j = r.json()
            except Exception:
                logger.error("explore tags. error in parsing json")
                continue
        else:
            logger.warn("explore tags. status code: %d", r.status_code)
            continue
        nodes = j['graphql']['hashtag']['edge_hashtag_to_media']['edges']
        captions = [_extract_caption(n) for n in nodes]
        for caption in captions:
            normalized_tags += tags_from_caption(caption)
    counter = Counter(normalized_tags)
    for target_tag in target_tags:
        try:
            counter.pop('#' + target_tag)
        except KeyError:
            logger.error("hashtag %s should be in dict but not", target_tag)
    counted_tags = list(counter.items())
    counted_tags.sort(key=lambda x: x[1], reverse=True)
    return counted_tags


def all_related(tag, blacklist=[]):
    return [x[0] for x in _related_tags(tag)]


def top_related(tags, k, blacklist=[]):
    return [x[0] for x in _related_tags(*tags)[:k]]


if __name__ == '__main__':
    tags = sys.argv[1:]
    print(top_related(tags, 15))
