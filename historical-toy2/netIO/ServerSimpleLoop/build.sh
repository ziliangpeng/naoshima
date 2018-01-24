#!/bin/bash

cp ../net_common.py ./
cp ../server_simple_loop.py ./

docker build -t netio/server_simple_loop .

