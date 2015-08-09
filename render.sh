#!/bin/bash
FILE=$1
time python videosonify.py "$FILE"
mencoder -ovc copy -audiofile out.wav -oac copy "$FILE" -o "`basename \"$FILE\"`-deeplearn$2.mkv"
python out-2-csound.py > grain.sco
csound -o output.wav grain.orc grain.sco
# mplayer -audiofile output.wav *mkv
mencoder -ovc copy -audiofile output.wav -oac copy "$FILE" -o "`basename \"$FILE\"`-granular-deeplearn$2.mkv"
