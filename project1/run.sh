#!/bin/bash

if [ ! -d output/figures ]; then
  mkdir -p output/figures;
  echo "created folder: output/figures"
fi

FILE1=datafiles/SRTM_data_Norway_1.tif
FILE2=datafiles/SRTM_data_Norway_2.tif

if [ -f "$FILE1" -a -f "$FILE2" ]; then
  echo "  "
else
  echo "Datafiles: $FILE1 and $FILE2 does not exist"
fi
