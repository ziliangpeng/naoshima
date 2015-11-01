#!/bin/bash

cd ServerSimpleLoop
./build.sh >/dev/null
cd ..

cd ServerSimpleThreading
./build.sh >/dev/null
cd ..

cd ClientSimpleLoop
./build.sh >/dev/null
cd ..


# Start all servers

docker run -d --name server_simple_loop netio/server_simple_loop
docker run -d --name server_simple_threading netio/server_simple_threading


time docker run -it --rm --link server_simple_loop:server netio/client_simple_loop 
time docker run -it --rm --link server_simple_threading:server netio/client_simple_loop 
time docker run -it --rm --link server_simple_loop:server netio/client_simple_loop_bg
time docker run -it --rm --link server_simple_threading:server netio/client_simple_loop_bg

docker rm -f server_simple_loop >/dev/null
docker rm -f server_simple_threading >/dev/null
