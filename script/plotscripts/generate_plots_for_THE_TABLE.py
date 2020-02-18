import sys, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

root_path = sys.argv[1]

BCST = "BehaviorCloning_State_FullTrajectory"
BCSS = "BehaviorCloning_State_State"
GST = "GAIL_State_FullTrajectory"
GSS = "GAIL_State_State"

EXPS = [BCST, BCSS, GST, GSS]

def parse_log(exp_log):
    exp_log = open(root_path + "/" + exp_log + ".txt").readlines()[1:]
    exp_log = np.array([[float(num) for num in line.split()] for line in exp_log])[:,[0,-2,-1]]
    return exp_log

exp_data = {
    name: parse_log(name) for name in EXPS
}

exp_data[BCST][:,0] *= 10000
exp_data[BCSS][:,0] *= 10000


exp_data[GST][:,0] *= 128
exp_data[GSS][:,0] *= 128

print exp_data









sys.exit()


def load_log_files_under_path(path):
    all_data = []
    legend_names = []
    legends = []
    for i,dirs in enumerate(sorted(os.listdir(path))):
        print dirs
        fname = "train_log.txt"
        abs_fname = path + "/" + dirs + "/" + fname
        if not os.path.exists(abs_fname):
            continue
        print "processing ",abs_fname
        folder_desc = dirs.split("_")
        legend_name = []
        legend = []
        with open(abs_fname) as train_log:
            lines = train_log.readlines()
            header = lines[0]
            real_datasize = header.split(" ")[-1]
            print header, real_datasize
            for j,val in enumerate(folder_desc):
                if j%2 is not 0:
                    continue
                legend_name.append(folder_desc[j])
                if folder_desc[j] == 'data':
                    legend.append(int(real_datasize))
                else:
                    legend.append(int(folder_desc[j+1]))

            lines = lines[1:]
            data = np.array([[float(num) for num in line.split()] for line in lines]).transpose()

            all_data.append(data)
            legend_names.append(legend_name)
            legends.append(legend)
    assert all(x == legend_names[0] for x in legend_names)
    return np.array(all_data), np.array(legend_names[0]), np.array(legends)
        

root_path = sys.argv[1]

data, legend_names, legends = load_log_files_under_path(root_path)

#for i in range(9):
#    plot(data, i+1)
print data.shape
print legend_names
print legends
print

fig, ax = plt.subplots()
plt.legend(loc='best')
data_names = "itera, Discriminator_Training_Loss, Discriminator_Training_Accuracy, Generator_Training_Loss, g_train_MAE, Discriminator_Eval_Loss, Discriminator_Eval_Accuracy, Generator_Eval_Loss, Generator_Eval_Accuracy, Min_MSE_Train, Min_MSE_Eval".replace(" ", "").split(",")

critera = 9
C = critera
plt.xlabel('Trained_Iterations')
plt.ylabel('MinOfMeanSquareErrors')
ax.set_ylim([0,0.5])


for i in range(legends.shape[0]):
    ax.plot(data[i][0], data[i][C], label = ("DataSize = %d Train" % legends[i][0]))

cmap = plt.cm.get_cmap('Blues')
rang = [0.1, 1]
colors = [cmap(i) for i in np.linspace(rang[0], rang[1], legends.shape[0])]
plt.gca().set_color_cycle(colors)

save_fname = 'the_table.png'
plt.legend(loc='best')
plt.savefig("./pngs/" + save_fname)
plt.show()


#################################################################
