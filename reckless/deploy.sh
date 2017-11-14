#!/bin/bash

# build
docker build -t reckless .

# stop running instance
echo "running containers:"
docker ps -a -q  --filter ancestor=reckless
docker ps -a -q  --filter ancestor=reckless | xargs docker kill
echo "existing containers:"
docker ps -a -q  --filter ancestor=reckless
docker ps -a -q  --filter ancestor=reckless | xargs docker rm

# wait
sleep 2
# run new deployment
docker run -it -d -p 80:80 reckless
