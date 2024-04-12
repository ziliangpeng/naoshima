#!/bin/bash

clang mmap-benchmark.c

./a.out ./files/1m.dat
./a.out ./files/10m.dat
./a.out ./files/100m.dat
./a.out ./files/1g.dat