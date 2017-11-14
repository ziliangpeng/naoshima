#!/bin/zsh
# This script uses some local aliased commands. Not supposed to use elsewhere.
source ~/dotfiles/zsh/local.zsh

ins-stats > /tmp/stat.txt 2>/dev/null
ins-eva-stats > /tmp/eva-stat.txt 2>/dev/null
ins-z-stats > /tmp/z-stat.txt 2>/dev/null
ins-newdone-stats > /tmp/newdone-stat.txt 2>/dev/null
ins-nesign-stats > /tmp/nesign-stat.txt 2>/dev/null
ins-hixchan-stats > /tmp/hixchan-stat.txt 2>/dev/null

python plot-all.py /tmp/stat.txt /tmp/eva-stat.txt /tmp/z-stat.txt /tmp/newdone-stat.txt /tmp/nesign-stat.txt /tmp/hixchan-stat.txt
