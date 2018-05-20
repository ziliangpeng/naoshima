#!/usr/bin/env bash


docker build -t test-general .

docker run -e ENV_NAME=V test-general ./d-print-env.sh
