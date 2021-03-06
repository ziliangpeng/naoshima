#!/bin/bash
set -e  # exit on error
set -x  # print command before execution


# Usage:
#
# ./merge-repo.sh old_repo main_repo



GITHUB_USERNAME="ziliangpeng"
A_REPO="https://github.com/$GITHUB_USERNAME/$1.git"
A_DIR="/tmp/a_repo_$1"
A_ALIAS_DIR="historical/historical-$1"
git clone $A_REPO $A_DIR
cd $A_DIR
git remote rm origin
git filter-branch --subdirectory-filter $A_ALIAS_DIR -- --all && true
mkdir -p $A_ALIAS_DIR
mv * $A_ALIAS_DIR && true
mv .gitignore $A_ALIAS_DIR
git add .
git commit -m "[git] move historical repo $1 into subdir"


B_REPO="git@github.com:$GITHUB_USERNAME/$2.git"
B_DIR="/tmp/b_repo"

git clone $B_REPO $B_DIR
cd $B_DIR
git checkout -b "v--migrate-$1"
git remote add repo-A-branch $A_DIR
git pull repo-A-branch master --allow-unrelated-histories
git remote rm repo-A-branch


