#!/bin/bash

INPUT=$1
OUTPUT=$2
MODEL=$3
START=${4-0}
DUR=${5-0}
FILENAME="${1##*/}"
FILENAME="${FILENAME%.*}"
FPS=`ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 $INPUT`

mkdir -p frames
echo "Extracting frames"
ffmpeg -ss $START -t $DUR -i $INPUT frames/$FILENAME%d.png
echo "Finished extracting frames, transforming"
for f in frames/$FILENAME*.png; do python generate.py $f -m $MODEL -o frames/trans_${f:7} -g 0; done
echo "Done processing, muxing back togeter"
ffmpeg -framerate $FPS -i frames/trans_$FILENAME%d.png -c:v libx264 $OUTPUT