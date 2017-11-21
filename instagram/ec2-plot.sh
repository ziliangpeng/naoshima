#!/bin/bash

python3 plot-likes.py \
    /data/ig/v_likes.txt \
    /data/ig/eva_likes.txt \
    /data/ig/z_likes.txt \
    /data/ig/newdone_likes.txt \
    /data/ig/nesign_likes.txt \
    /data/ig/hixchan_likes.txt

python3 plot-all.py \
    /data/ig/v_count.txt \
    /data/ig/eva_count.txt \
    /data/ig/z_count.txt \
    /data/ig/newdone_count.txt \
    /data/ig/nesign_count.txt \
    /data/ig/hixchan_count.txt
