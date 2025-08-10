#!/bin/bash
mkdir games
cd games
wget https://badukmovies.com/pro_games/download --no-check-certificate --no-check-certificate
unzip download
rm download
cd ..
python build_dataset.py
python train.py
