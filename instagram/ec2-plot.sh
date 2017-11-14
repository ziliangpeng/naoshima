#!/bin/bash

python3 plot-likes.py \
    ~/instagram_likes.txt \
    ~/eva_instagram_likes.txt \
    ~/z_instagram_likes.txt \
    ~/newdone_instagram_likes.txt \
    ~/nesign_instagram_likes.txt \
    ~/hixchan_instagram_likes.txt

python3 plot-all.py \
    ~/instagram_followers_count.txt \
    ~/eva_instagram_followers_count.txt \
    ~/z_instagram_followers_count.txt \
    ~/newdone_instagram_followers_count.txt \
    ~/nesign_instagram_followers_count.txt \
    ~/hixchan_instagram_followers_count.txt
