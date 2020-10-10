#!/bin/bash

if [ ! -d output/figures ]; then
  mkdir -p output/figures;
  echo "created folder: output/figures"
fi

if [ ! -d output/outfiles ]; then
  mkdir -p output/outfiles;
  echo "created folder: output/outfiles"
fi

python main_ridge.py
