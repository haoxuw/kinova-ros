import sys, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_log_files_under_path(path):
    all_data = []
    legend_names = []
    legends = []
    for i,dirs in enumerate(sorted(os.listdir(path))):
        fname = "train_log.txt"
        abs_fname = path + "/" + dirs + "/" + fname
        if not os.path.exists(abs_fname):
            continue
        print "processing ",abs_fname
        header = dirs.split("_")
        legend_name = []
        legend = []
        for j,val in enumerate(header):
            if j%2 is not 0:
                continue
            legend_name.append(header[j])
            legend.append(int(header[j+1]))

        with open(abs_fname) as train_log:
            lines = train_log.readlines()
            print lines[0]
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
cmap = plt.cm.get_cmap('Oranges')
rang = [0.1, 1]
colors = [cmap(i) for i in np.linspace(rang[0], rang[1], legends.shape[0])]
plt.gca().set_color_cycle(colors)

data_names = "itera, Discriminator_Training_Loss, Discriminator_Training_Accuracy, Generator_Training_Loss, g_train_MAE, Discriminator_Eval_Loss, Discriminator_Eval_Accuracy, Generator_Eval_Loss, Generator_Eval_Accuracy, Min_MSE_Train, Min_MSE_Eval".replace(" ", "").split(",")

critera = 9
C = critera
plt.xlabel('Trained_Iterations')
plt.ylabel(data_names[C])
ax.set_ylim([0,0.5])

for i in range(legends.shape[0]):
    ax.plot(data[i][0], data[i][C], label = ("Experiment # %d Train" % i))

cmap = plt.cm.get_cmap('Blues')
rang = [0.1, 1]
colors = [cmap(i) for i in np.linspace(rang[0], rang[1], legends.shape[0])]
plt.gca().set_color_cycle(colors)

plt.ylabel(data_names[C])
ax.set_ylim([0,0.5])
C += 1
for i in range(legends.shape[0]):
    ax.plot(data[i][0], data[i][C], label = ("Experiment # %d Eval" % i))

save_fname = 'exp.png'
plt.legend(loc='best')
plt.savefig("../pngs/" + save_fname)


