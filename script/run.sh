#!/bin/bash

#set -x
set -e
unit_size=1000
max_iter=100

python learn_generator.py --max_size -1 --itera 0

for (( cnt=1; cnt <= max_iter; cnt++ ))
do
    echo ${cnt}
    (( size=unit_size * cnt ))
    echo ${size}
    python learn_generator.py --max_size ${size} --re_train --itera ${cnt} || play -n synth 1
done

play -n synth 1
