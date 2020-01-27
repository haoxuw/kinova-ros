import sys, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_log_files_under_path(path):
    all_data = {}
    for i,dirs in enumerate(sorted(os.listdir(path))):
        fname = "train_log.txt"
        abs_fname = path + "/" + dirs + "/" + fname
        if not os.path.exists(abs_fname):
            continue
        print abs_fname
        criteria_arr = dirs.split("_")
        for i in range(len(criteria_arr)):
            if criteria_arr[i] == "epoch":
                criteria = criteria_arr[i+1]

        with open(abs_fname) as train_log:
            lines = train_log.readlines()[1:]
            data = np.array([[float(num) for num in line.split()] for line in lines]).transpose()

            all_data[int(criteria)] = data
    return all_data
        

def plot(all_data, plot_index = 1):
    legend = "itera, fit_loss_discriminator, d_train_accuracy, fit_loss_generator, g_train_MAE, eval_loss_discriminator, eval_accuracy_discriminator, eval_loss_generator, eval_accuracy_generator, traj_eval_MSE".replace(" ", "").split(",")


    print legend

    fig, ax = plt.subplots()

    
    keys = sorted([int(x) for x in data.keys()])
    cmap = plt.cm.get_cmap('Reds')

    rang = [0.2, 1]
    colors = [cmap(i) for i in np.linspace(rang[0], rang[1], len(keys))]
    plt.gca().set_color_cycle(colors)


    for key in keys:
        print key
        value = data[key]
        itera, d_train_loss, d_train_accuracy, g_train_loss, g_train_MAE, d_eval_loss, d_eval_accuracy, g_eval_loss, g_eval_MAE, traj_eval_MSE = value
        ax.plot(value[0], value[plot_index], label = ("Epochs = %s" % key))
        #ax.axis('legend[index]')

    plt.legend(loc='best')
    plt.xlabel('Training iterations')
    plt.ylabel(legend[plot_index])
    filename = "epochs"
    plt.savefig(filename + '.png', dpi=600)

    plt.show()


data = load_log_files_under_path(sys.argv[1])

for i in range(9):
    plot(data, i+1)

