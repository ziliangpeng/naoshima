#!/bin/bash

OUTPUT_RESOLUTION=$1

function convert_to {
  filename=$1
  res=$2
  echo ">>>Starting" `date`
  echo "converting '$filename' to $res"

  tmp_file=$filename-converted-to-$res-tmp.mov
  target_file=$filename-converted-to-$res.mov
  stdout_file=$filename-converted-to-$res.out
  stderr_file=$filename-converted-to-$res.err

  if [ -f "$target_file" ]; then
    echo "skipping..."
  else
    echo "working..."
    echo `date` >>"$stdout_file"
    echo `date` >>"$stderr_file"

    ffmpeg -i "$filename" -vf scale=-1:$res "$tmp_file" 1>>"$stdout_file" 2>>"$stderr_file"
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
    convert_to "$file" $OUTPUT_RESOLUTION >> $log_file
    if [ $? -ne 0 ]; then
      exit 1
    fi
  fi
done
