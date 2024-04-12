#!/bin/bash

# For macOS

mkdir -p ./files

dd if=/dev/urandom bs=1M count=1 of=./files/1m.dat
dd if=/dev/urandom bs=1M count=10 of=./files/10m.dat
dd if=/dev/urandom bs=1M count=100 of=./files/100m.dat
dd if=/dev/urandom bs=1M count=1000 of=./files/1g.dat
