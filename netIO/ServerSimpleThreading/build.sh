#!/bin/bash

cp ../net_common.py ./
cp ../server_simple_threading.py ./

docker build -t netio/server_simple_threading .

