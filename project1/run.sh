#!/bin/bash

if [ ! -d output/figures ]; then
  mkdir -p output/figures;
  echo "created folder: output/figures"
fi

if [ ! -d output/outfiles ]; then
  mkdir -p output/outfiles;
  echo "created folder: output/outfiles"
fi

python main.py

#echo -n "Do you want to run a) [y/n]?"
#read type

#if [ "$type" == "y" ]; then
#  echo "Running part a)"
#  echo "--------------------------------------------------"
#  python main.py 0
#fi
#
#if [ "$type" == "n" ]; then
#  echo "Skipping part a)"
#  echo "--------------------------------------------------"
#  python main.py 1
#fi



#FILE1=datafiles/SRTM_data_Norway_1.tif
#FILE2=datafiles/SRTM_data_Norway_2.tif

#if [ -f "$FILE1" -a -f "$FILE2" ]; then
#  echo "  "
#  python main.py
#else
#  echo "Datafiles: $FILE1 and $FILE2 does not exist"
#  echi "still want to"
#fi
