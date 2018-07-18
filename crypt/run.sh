#!/usr/bin/env bash

docker build -t crypt .

docker run crypt
