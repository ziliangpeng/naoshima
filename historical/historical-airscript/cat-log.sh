#!/bin/zsh

HOST=$1
for i in `seq 1 100`; time ssh $1.inst.aws.airbnb.com cat /var/log/init.err > /dev/null

