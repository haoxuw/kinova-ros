#!/bin/bash

#set -x
set -e
max_size=100000 #0
unit_size=1000
max_iter=1000

epochs=16
rm -rf final_results/NN_model*

python learn_generator.py --max_size ${max_size} --itera 0 --epochs ${epochs} --batch 16

#exit

for (( cnt=1; cnt <= max_iter; cnt++ ))
do
    (( size=unit_size * cnt ))
    python learn_generator.py --max_size ${size} --epochs 2 --batch 32 --re_train --itera ${cnt} # || play -n synth 1

done

play -n synth 1
