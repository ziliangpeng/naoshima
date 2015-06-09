#!/bin/bash

cd ServerSimpleLoop
./build.sh
cd ..

cd ClientSimpleLoop
./build.sh
cd ..


docker run -d --name server_simple_loop netio/server_simple_loop

time docker run -it --rm --link server_simple_loop:server netio/client_simple_loop 


docker rm -f server_simple_loop
