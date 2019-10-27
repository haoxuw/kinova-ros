#!/bin/bash


#set -x
set -e
#data_size=1000
pre_train_size=10 #000
max_iter=3201

epochs=4
batch=64 #16 52
#rm -rf final_results/*
for data_size in 10000 10 100 1000 100000 1000000
do
    for epochs in  16 8 4 2 1 32 # 
    do
	for exp in 1 #4 5 6 7 8
	do
	    python process_scripts.py --max_size ${data_size} # max_size = total_data_size
	    rm final_results/NN_model_* -rf && python learn_generator.py --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
	    python learn_generator.py --epochs ${epochs} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_exp_${exp}_epoch_${epochs}_batch_${batch} #  || play -n synth 1
	done
    done
done









exit

for data_size in 100 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000
do
    python process_scripts.py --max_size ${data_size}
    python learn_generator.py --max_size ${data_size} --train_bc --epochs 32 --save_fig_folder bc_data_${data_size}
done

exit

