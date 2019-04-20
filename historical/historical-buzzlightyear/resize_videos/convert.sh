#!/bin/bash

SPEED=$1
QUALITY=$2

function convert_to {
  filename=$1

  echo ">>>Starting" `date`
  echo "converting '$filename'"

  target_file=$filename-converted-to-$SPEED-$QUALITY.mov
  tmp_file=$target_file.tmp.mov
  stdout_file=$target_file.out
  stderr_file=$target_file.err

  if [ -f "$target_file" ]; then
    echo "skipping..."
  else
    echo "working..."
    echo `date` >>"$stdout_file"
    echo `date` >>"$stderr_file"

    # following can preserve modified time, but seems only work at linux
    # -metadata date="$(stat --printf='%y' inputfile.mp4 | cut -d ' ' -f1)"
    ffmpeg -i "$filename" -preset $SPEED -crf $QUALITY "$tmp_file" 1>>"$stdout_file" 2>>"$stderr_file"
    if [ $? -ne 0 ]; then
      echo "is interrupted."
      return 1
    else
      mv "$tmp_file" "$target_file"
    fi
  fi
  echo ">>>>>Ending" `date` "$filename"
}

log_file="./convert.log"
echo "======= START =======" >> $log_file

for file in "."/*.mov
do
  echo $file | grep "converted" > /dev/null
  if [ $? -ne 0 ]; then
    convert_to "$file" >> $log_file
    if [ $? -ne 0 ]; then
      exit 1
    fi
  fi
done
