#!/bin/bash

python plot-likes.py \
    /data/ig/v_likes.txt \
    /data/ig/eva_likes.txt \
    /data/ig/z_likes.txt \
    /data/ig/newdone_likes.txt \
    /data/ig/hixchan_likes.txt \
    /data/ig/mc_likes.txt \
    /data/ig/let_likes.txt \
    /data/ig/b1_likes.txt \
    /data/ig/eaxy_likes.txt

python plot-all.py \
    /data/ig/v_count.txt \
    /data/ig/eva_count.txt \
    /data/ig/z_count.txt \
    /data/ig/newdone_count.txt \
    /data/ig/hixchan_count.txt \
    /data/ig/mc_count.txt \
    /data/ig/let_count.txt \
    /data/ig/b1_count.txt \
    /data/ig/eaxy_count.txt
