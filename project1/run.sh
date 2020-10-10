#!/bin/bash

if [ ! -d output/figures/franke ]; then
  mkdir -p output/figures/franke;
  echo "created folder: output/figures/franke"
fi

if [ ! -d output/figures/data ]; then
  mkdir -p output/figures/data;
  echo "created folder: output/figures/data"
fi

echo "Do you want to run on the Franke Function [f] or real data [d]?"
read type

if [ "$type" == "f" ]; then
  echo "Do you want OLS [o], RIDGE [r], LASSO [l] or all [a]?"
  read rtype
  if [ "$rtype" == "o" ]; then
    python main_ols.py franke 
  fi
  if [ "$rtype" == "r" ]; then
    python main_ridge.py franke
  fi
  if [ "$rtype" == "l" ]; then
    python main_lasso.py franke
  fi
  if [ "$rtype" == "a" ]; then
    python main_ols.py franke
    python main_ridge.py franke
    python main_lasso.py franke
  fi
fi

if [ "$type" == "d" ]; then
  echo "Do you want OLS [o], RIDGE [r], LASSO [l] or all [a]?"
  read rtype
  if [ "$rtype" == "o" ]; then
    python main_ols.py data
  fi
  if [ "$rtype" == "r" ]; then
    python main_ridge.py data
  fi
  if [ "$rtype" == "l" ]; then
    python main_lasso.py data
  fi
  if [ "$rtype" == "a" ]; then
    python main_ols.py data
    python main_ridge.py data
    python main_lasso.py data
  fi
fi
