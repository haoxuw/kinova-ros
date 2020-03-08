#!/bin/bash


#set -x
#data_size=1000
pre_train_size=10 #000
max_iter=6401
#max_iter=101
set -x
name="pouring"
epochs=1
batch=128 #16 52
#rm -rf ${name}_results/*

for data_size in 300 # 1000 100 # 10000 # 
do
    max_iter=101
    python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for gen_steps in 1 # 10
    do
	for epochs in 1 # 8 16
	do
	    for exp in 1 # 2 3 4 5 6
	    do
		python learn_generator.py --state_to_action --name ${name} --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
		python learn_generator.py --state_to_action --name ${name} --epochs ${epochs} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_exp_${exp}_epoch_${epochs}_batch_${batch}_STS #  || play -n synth 1
		python learn_generator.py --name ${name} --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
		python learn_generator.py --name ${name} --epochs ${epochs} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_dissteps_${dis_steps}_epoch_${epochs}_batch_${batch}_exp_${exp} #  || play -n synth 1
	    done
	done
    done
done


exit

#################################################
## BC state to full traj
#################################################

for data_size in 10000 #1000 100 100000 1000000 # 10000
do
    python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for epochs in 110
    do
	for exp in 1 #4 5 6 7 8
	do
	    rm ${name}_results/NN_model_* -rf
	    python learn_generator.py --name ${name} --epochs ${epochs} --batch ${batch} --train_bc --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_exp_${exp}_epoch_${epochs}_batch_${batch}_BC_STTRAJ #  || play -n synth 1
	done
    done
done

exit

#################################################
## BC state to action
#################################################

for data_size in 7475 #1000 100 100000 1000000 # 10000
do
    #python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for epochs in 110
    do
	for exp in 1 #4 5 6 7 8
	do
	    rm ${name}_results/NN_model_* -rf
	    python learn_generator.py --name ${name} --epochs ${epochs} --batch ${batch} --train_bc --state_to_action --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_exp_${exp}_epoch_${epochs}_batch_${batch}_BC_STACTION #  || play -n synth 1
	done
    done
done

#################################################
## GAIL state to action
#################################################
for data_size in 30000 # 1000 100 # 10000 # 
do
    #python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for gen_steps in 1 # 10
    do
	for epochs in 1 # 8 16
	do
	    for exp in 1 # 2 3 4 5 6
	    do
		rm ${name}_results/NN_model_* -rf
		python learn_generator.py --state_to_action --name ${name} --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
		python learn_generator.py --state_to_action --name ${name} --epochs ${epochs} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_exp_${exp}_epoch_${epochs}_batch_${batch}_STS #  || play -n synth 1
	    done
	done
    done
done


exit

for data_size in 30000 # 1000 100 # 10000 # 
do
    #python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for gen_steps in 1 # 10
    do
	for epochs in 1 # 8 16
	do
	    for exp in 1 2 3 4 5 6
	    do
		rm ${name}_results/NN_model_* -rf
		python learn_generator.py --name ${name} --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
		python learn_generator.py --name ${name} --epochs ${epochs} --gen_steps ${gen_steps} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_gensteps_${gen_steps}_epoch_${epochs}_batch_${batch}_exp_${exp} #  || play -n synth 1
	    done
	done
    done
done

for data_size in 100000 # 1000 100 # 10000 # 
do
    python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for dis_steps in 4 2
    do
	for epochs in 1 2 # 8 16
	do
	    for exp in 1 2
	    do
		rm ${name}_results/NN_model_* -rf
		python learn_generator.py --name ${name} --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
		python learn_generator.py --name ${name} --epochs ${epochs} --dis_steps ${dis_steps} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_dissteps_${dis_steps}_epoch_${epochs}_batch_${batch}_exp_${exp} #  || play -n synth 1
	    done
	done
    done
done



exit

#set -e

exit



#################################################
## data size
#################################################

for data_size in 30000 # 1000 300 100 100000 10000 3000
do
    rm ${name}_results/*.npy
    python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for gen_steps in 1 # 10
    do
	for epochs in 1 # 8 16
	do
	    for exp in 1 # 2 3 4 5 6
	    do
		rm ${name}_results/NN_model_* -rf
		python learn_generator.py --name ${name} --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
		python learn_generator.py --name ${name} --epochs ${epochs} --gen_steps ${gen_steps} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_gensteps_${gen_steps}_epoch_${epochs}_batch_${batch}_exp_${exp} #  || play -n synth 1
	    done
	done
    done
done

for data_size in 30000 # 1000 100 # 10000 # 
do
    #python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for gen_steps in 1 2 4 8
    do
	for epochs in 1 # 8 16
	do
	    for exp in 1
	    do
		rm ${name}_results/NN_model_* -rf
		python learn_generator.py --name ${name} --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
		python learn_generator.py --name ${name} --epochs ${epochs} --gen_steps ${gen_steps} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_gensteps_${gen_steps}_epoch_${epochs}_batch_${batch}_exp_${exp} #  || play -n synth 1
	    done
	done
    done
done

for data_size in 30000 # 1000 100 # 10000 # 
do
    #python process_scripts.py --max_size ${data_size} # max_size = total_data_size
    for dis_steps in 1 2 4 8
    do
	for epochs in 1 # 8 16
	do
	    for exp in 1
	    do
		rm ${name}_results/NN_model_* -rf
		python learn_generator.py --name ${name} --max_size ${pre_train_size} --epochs 1 --batch 16 --init_gan # max_size = pre_train size
		python learn_generator.py --name ${name} --epochs ${epochs} --dis_steps ${dis_steps} --batch ${batch} --train_gan --itera ${max_iter} --save_fig_num 9 --save_fig_folder data_${data_size}_pretrain_${pre_train_size}_dissteps_${dis_steps}_epoch_${epochs}_batch_${batch}_exp_${exp} #  || play -n synth 1
	    done
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

