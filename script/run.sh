#!/bin/bash

#set -x
set -e
#data_size=1000
pre_train_size=1000
max_iter=7000

epochs=2
#rm -rf final_results/*
for data_size in 10000 100 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000
do
    #python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    python learn_generator.py --max_size ${pre_train_size} --epochs ${epochs} --batch 16 # max_size = pre_train size
    python learn_generator.py --epochs 16 --batch 512 --train_gan --itera ${max_iter} --save_fig_num 5 --save_fig_folder data_${data_size}_pretrain_${pre_train_size} #  || play -n synth 1
    echo "Looping ... number $i"
done

