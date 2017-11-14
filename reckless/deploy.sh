#!/bin/bash

# build
docker build -t reckless .

# stop running instance
docker kill reckless-prod

# wait
sleep 2

# run new deployment
docker run -d --rm --name=reckless-prod -v "$(pwd)"/static:/reckless/static -p 80:80 reckless
