#!/usr/bin/env bash

docker build -q -t crypt .

docker run crypt $@
