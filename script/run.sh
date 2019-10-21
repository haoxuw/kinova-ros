#!/bin/bash

#set -x
set -e
#data_size=1000
pre_train_size=10 #000
max_iter=3200

epochs=4
rm -rf final_results/*
for data_size in 10000 100 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000
do
    for exp in 1 2 # 3 4 5 6 7 8
    do
	python process_scripts.py --max_size ${data_size} # max_size = total_data_size
	rm final_results/NN_model_* -rf && python learn_generator.py --max_size ${pre_train_size} --epochs ${epochs} --batch 16 --init_gan # max_size = pre_train size
	python learn_generator.py --epochs 16 --batch 52 --train_gan --itera ${max_iter} --save_fig_num 5 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_exp_${exp} #  || play -n synth 1
	echo "Looping ... number $i"
    done
done

