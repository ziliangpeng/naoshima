#!/bin/bash

cd ~/.ssh

cat known_hosts | grep -v "inst.aws.airbnb.com" > filtered_hosts

cp known_hosts known_hosts.`date +"%Y%m%d"`.bak

mv filtered_hosts known_hosts
