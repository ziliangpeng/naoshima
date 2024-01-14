import requests
import re
import statsd

import leveldb


ARKB_URL = "https://ark-funds.com/funds/arkb/"

def get_arkb():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'Dnt': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    session = requests.Session()
    a = session.get(ARKB_URL, headers=headers)
    return a.status_code, a.text

def get_na(text): # Net Assets
    r = "<li>Net Assets <span>\$([\d.,]+) Million<\/span><\/li>"
    match = re.search(r, text)
    if match.groups(): return match.group(1)
    else: return 0

def write2db():
    pass

code, text = get_arkb()
na = get_na(text)

s = statsd.StatsClient("localhost", 8125)
s.gauge("arkb.netasserts", float(na))

