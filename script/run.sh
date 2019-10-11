#!/bin/bash

#set -x
set -e
max_size=10000
unit_size=1000
max_iter=1000

epochs=2
rm -rf final_results/*
python process_scripts.py --max_size ${max_size} # max_size = total_data_size
python learn_generator.py --max_size ${max_size} --epochs ${epochs} --batch 16 # max_size = pre_train size
python learn_generator.py --epochs 4 --batch 32 --train_gan --itera 10000 --save_fig_num 5 #  || play -n synth 1

