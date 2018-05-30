#!/bin/bash

branch=`git rev-parse --abbrev-ref HEAD`
commits=`git log --reverse --pretty=format:"%H"`

for i in $commits
do
  git checkout $i 
done

git checkout $branch
