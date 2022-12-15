#!/bin/bash

for ts in 16
do
for heads in 8 16 
do
for depth in 3
do
nohup python3 htransformer1d_script.py --token_size $ts --heads $heads --depth $depth > log$heads.txt&
sleep 2
done
done 
done

# females TNT
# 8 depth 3
#16 depth 3
#32 depth 2

# Mancano TNT:
# - females 32x3
# - males 8x3