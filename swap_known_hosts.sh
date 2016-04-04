#!/bin/bash

cd ~/.ssh

cat known_hosts | grep -v "inst.aws.airbnb.com" > filtered_hosts

mv filtered_hosts known_hosts
