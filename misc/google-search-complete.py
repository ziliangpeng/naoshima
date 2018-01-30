#!/bin/env python3
import itertools
import random

import requests
import time
import argparse


def get_completion(keyword):
    URL = 'https://www.google.com/complete/search?gs_ri=psy-ab&q=' + keyword
    r = requests.get(URL)
    j = r.json()
    print(keyword, [match[0] for match in j[1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test")
    parser.add_argument("--key", type=str)
    args = parser.parse_args()

    if args.key:
        k = args.key
        get_completion(k)
    elif args.test:
        start_time = time.time()
        for i in itertools.count():
            print(i)
            print(time.time() - start_time)
            k = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(4)])
            get_completion(k)