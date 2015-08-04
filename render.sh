#!/bin/bash
FILE=$1
time python videosonify.py "$FILE"
mencoder -ovc copy -audiofile out.wav -oac copy "$FILE" -o "`basename \"$FILE\"`-deeplearn$2.mkv"
