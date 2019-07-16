#!/bin/bash
python3 ../main.py --dir /data2/hhjung/Normalization/P2/group4/batch16 -b 16 --Method P2 --group 4 --gpu 7 --layer 2 1> /dev/null && 
python3 ../main.py --dir /data2/hhjung/Normalization/P2/group4/batch32 -b 32 --Method P2 --group 4 --gpu 7 --layer 2 1> /dev/null && 
python3 ../main.py --dir /data2/hhjung/Normalization/P2/group4/batch64 -b 64 --Method P2 --group 4 --gpu 7 --layer 2 1> /dev/null && 
python3 ../main.py --dir /data2/hhjung/Normalization/P2/group4/batch128 -b 128 --Method P2 --group 4 --gpu 7 --layer 2 1> /dev/null
