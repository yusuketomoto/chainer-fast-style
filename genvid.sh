#!/bin/bash

INPUT=$1
OUTPUT=$2
MODEL=$3
START=${4-0}
DUR=${5-0}
DIR=frames
FILENAME="${INPUT##*/}"
FILENAME="${FILENAME%.*}"
FPS=`ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 $INPUT`

mkdir -p $DIR
echo "Extracting frames"
ffmpeg -ss $START -t $DUR -i $INPUT $DIR/$FILENAME%d.png
echo "Finished extracting frames, transforming"
python generate.py $DIR/$FILENAME'*.png' -m $MODEL -o $DIR/trans_ -g 0
echo "Done processing, muxing back togeter"
ffmpeg -framerate $FPS -i $DIR/trans_$FILENAME%d.png -c:v libx264 $OUTPUT
