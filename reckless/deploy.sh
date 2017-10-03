#!/bin/bash

# build
docker build -t reckless .

# stop running instance
docker ps -a -q  --filter ancestor=reckless | xargs docker kill
docker ps -a -q  --filter ancestor=reckless | xargs docker rm

# wait
sleep 2
# run new deployment
docker run -it -d -p 80:80 reckless
