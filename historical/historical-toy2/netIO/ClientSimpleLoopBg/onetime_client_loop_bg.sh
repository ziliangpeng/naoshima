#!/bin/bash

# this multi-processes approach cannot have too many processes running in parallel
for i in {1..500} 
do
    python onetime_client.py &
done
wait

