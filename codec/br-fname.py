#!/usr/bin/python3
"""Rename file to include bitrate in filename."""

import ffmpeg
import sys
import os

filename = sys.argv[1]

def get_bitrate(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise Exception('No video stream found')
    return video_stream['bit_rate']

br = get_bitrate(filename)

origin_prefix = filename
if origin_prefix.endswith('.mp4'):
    origin_prefix = origin_prefix[:-4]
new_filename = origin_prefix + '-br' + str(br) + '.mp4'
os.rename(filename, new_filename)
