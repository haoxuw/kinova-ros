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
plt.xlabel('Trained iterations')
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
plt.savefig("pngs/" + save_fname)


#################################################################



'''


def plot(data, plot_index = 1):
    legend = "itera, fit_loss_discriminator, d_train_accuracy, fit_loss_generator, g_train_MAE, eval_loss_discriminator, eval_accuracy_discriminator, eval_loss_generator, eval_accuracy_generator, traj_eval_MSE".replace(" ", "").split(",")

    legend = "itera, Discriminator_Training_Loss, Discriminator_Training_Accuracy, Generator_Training_Loss, g_train_MAE, Discriminator_Eval_Loss, Discriminator_Eval_Accuracy, Generator_Eval_Loss, Generator_Eval_Accuracy, traj_eval_MSE".replace(" ", "").split(",")
    #print legend

    fig, ax = plt.subplots()

    
    keys = sorted([int(x) for x in data.keys()])
    cmap = plt.cm.get_cmap('Oranges')

    rang = [0.1, 1]
    colors = [cmap(i) for i in np.linspace(rang[0], rang[1], len(keys))]
    plt.gca().set_color_cycle(colors)


    for key in keys:
        print key
        value = data[key]
        itera, d_train_loss, d_train_accuracy, g_train_loss, g_train_MAE, d_eval_loss, d_eval_accuracy, g_eval_loss, g_eval_MAE, traj_eval_MSE = value
        ax.plot(value[0], value[plot_index], label = ("Epochs = %s" % key))
        #ax.axis('legend[index]')

    plt.legend(loc='best')
    plt.xlabel('Trained iterations')
    plt.ylabel(legend[plot_index])
    filename = legend[plot_index]
    plt.savefig(root_path + '/' + filename + '.png', dpi=600)

    #plt.show()
    

#################################################################









#reward is calculated base
def calculate_reward():
    pass

def generate_plots_for_folder(abs_path, epochs):
    progresses = []
    for i in [0, 3, 6]:
        demo = '#0%d_Test_Demo.traj' %i
        abs_demo_fname = abs_path + '/' + demo
        if not os.path.exists(abs_demo_fname):
            print abs_demo_fname

        with open(abs_demo_fname) as demo_file:
            lines = demo_file.readlines()[1:]
            golden = np.array([[float(num) for num in line.split()] for line in lines])[:,1:]

            progress = []
            for j in range(0,6401,100):
                traj = '#0%d_Test_Predicted_In_Iteration_%d.traj' % (i,j)
                abs_traj_fname = abs_path + '/' + traj
                if not os.path.exists(abs_traj_fname):
                    print 'Warning, the following was not found: ' + abs_traj_fname
                    return

                with open(abs_traj_fname) as traj_file:
                    lines = traj_file.readlines()[1:]
                    experi = np.array([[float(num) for num in line.split()] for line in lines])[:,1:]

                    mse = ((golden - experi)**2).mean()
                    progress.append( [j, mse] )
            progress = np.array(progress).transpose()
            progresses.append(progress)

    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('Oranges')

    for i, progress in enumerate(progresses):
        ax.plot(progress[0], progress[1], label = ("Example #%d" % (i+1)))
        print progress.shape

    plt.legend(loc='best')
    plt.xlabel('Trained iterations')
    plt.ylabel('MeanSquredError_Reward')
    plt.title("Reward Progressions, Epochs = %s" % epochs)
    ax.set_ylim([0,0.2])

    filename = 'Rewards_Epochs_%s' % epochs
    plt.savefig(abs_path + '/' + filename + '.png', dpi=600)
    plt.savefig(abs_path + '/../' + filename + '.png', dpi=600)
    #plt.show()
            
    return

        
def load_log_files_under_path(path):
    all_data = {}
    for i,dirs in enumerate(sorted(os.listdir(path))):
        abs_path = path + "/" + dirs
        if not os.path.isdir(abs_path):
            continue
        if "data_" not in dirs:
            continue

        criteria_arr = dirs.split("_")
        for i in range(len(criteria_arr)):
            if criteria_arr[i] == "epoch":
                criteria = criteria_arr[i+1]
                
        generate_plots_for_folder(abs_path, criteria)




#################################################################


load_log_files_under_path(root_path)


#################################################################
'''
