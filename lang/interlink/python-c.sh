#!/bin/sh
gcc -shared -o clib.so -fPIC clib.c

python python-c.py
