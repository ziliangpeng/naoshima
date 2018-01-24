#!/bin/bash

proid="$1"

./run.sh "$proid" && echo `./exe/"$proid"` > "./ans/$proid.ans"
