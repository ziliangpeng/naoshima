#!/bin/bash


docker run -v /data/ardb:/var/lib/ardb -p 6379:16379 --name ardb-server -d lupino/ardb-server
