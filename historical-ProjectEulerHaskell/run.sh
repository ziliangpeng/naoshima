#!/bin/bash

proid="$1"

ghc --make -o ./exe/"$proid" "$proid".hs && ./exe/"$proid"
