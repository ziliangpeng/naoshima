#!/bin/bash

cp ../net_common.py ./
cp ../onetime_client.py ./
cp ../onetime_client_loop_bg.sh ./

docker build -t netio/client_simple_loop_bg .

