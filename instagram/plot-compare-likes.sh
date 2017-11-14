#!/bin/zsh
# This script uses some local aliased commands. Not supposed to use elsewhere.

source ~/dotfiles/zsh/local.zsh

ins-likes > /tmp/likes.txt 2>/dev/null
ins-eva-likes > /tmp/eva-likes.txt 2>/dev/null
ins-z-likes > /tmp/z-likes.txt 2>/dev/null
ins-newdone-likes > /tmp/newdone-likes.txt 2>/dev/null
ins-nesign-likes > /tmp/nesign-likes.txt 2>/dev/null
ins-hixchan-likes > /tmp/hixchan-likes.txt 2>/dev/null

python plot-likes.py /tmp/likes.txt /tmp/eva-likes.txt /tmp/z-likes.txt /tmp/newdone-likes.txt /tmp/nesign-likes.txt /tmp/hixchan-likes.txt
