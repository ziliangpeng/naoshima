#!/bin/bash

# build
docker build -t reckless .

# stop running instance
docker kill reckless-prod

# wait
sleep 2

# run new deployment
docker run -d --rm --name=reckless-prod -v /data/reckless:/reckless/static/data -p 80:80 reckless
