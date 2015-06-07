#!/bin/bash

for i in {1..100}
do
    python onetime_client.py &
done
wait

