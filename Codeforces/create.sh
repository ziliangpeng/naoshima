#!/bin/bash

# MSG='expected format: create -l python -f 42A'
MSG='expected format: create 42A python'

echo ''

if test $# -lt 2; then
	echo $MSG
	exit 1
fi

FILENAME=$1
LANGUAGE=$2

# while test $# -gt 0; do
# 	case "$1" in
# 		-l)
# 			shift
# 			if test $# -gt 0; then
# 				LANGUAGE=$1
# 			else
# 				echo $MSG
# 				exit 1
# 			fi
# 			shift
# 			;;

# 		-f)
# 			shift
# 			if test $# -gt 0; then
# 				FILENAME=$1
# 			else
# 				echo $MSG
# 				exit 1
# 			fi
# 			shift
# 			;;
# 		*)
# 			echo MSG
# 			exit 1
# 			;;
# 	esac
# done

if test -z "$LANGUAGE" || test -z "$FILENAME"; then
	echo $MSG
	exit 1
fi

FILENAME=$(echo $FILENAME | tr '[a-z]' '[A-Z]') # toupper
echo 'FILENAME IS:' $FILENAME

LANGUAGE=$(echo $LANGUAGE | tr '[A-Z]' '[a-z]') # tolower
case "$LANGUAGE" in
	python|py)
		suffix='py'
		;;
	cpp|c++)
		suffix='cpp'
		;;
esac
if test -z "$suffix"; then
	echo 'language '"$LANGUAGE"' not supported'
	exit 1
else
	echo 'LANGUAGE IS:' $suffix
fi

template='header.'$suffix
destination='tmp_'$FILENAME'.'$suffix
if test -f $destination; then
	echo 'file '$destination' already exist'
else
	echo 'copying '$template' to '$destination
	cp $template $destination
	echo 'done'
fi



