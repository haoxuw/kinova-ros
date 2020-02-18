import sys, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

assert len(sys.argv) == 2

log_filename = sys.argv[1]
output_name = 1#sys.argv[2]

def load_log(log_file):
    with open(log_file) as train_log:
        lines = train_log.readlines()
        print lines[0]
        lines = lines[1:]
        data = np.array([[float(num) for num in line.split()] for line in lines]).transpose()

    return np.array(data)

data = load_log(log_filename)

legend = "itera, Discriminator_Training_Loss, Discriminator_Training_Accuracy, Generator_Training_Loss, Generator_Training_Accuracy, Discriminator_Eval_Loss, Discriminator_Eval_Accuracy, Generator_Eval_Loss, Generator_Eval_Accuracy, Min_MSE_Train, Min_MSE_Eval".replace(" ", "").split(",")
legend = np.array(legend)

print data.shape
print legend

critera = [[1,3],[2,4],[5,7],[6,8],[5,10],[1,9]]

for pair in critera:
    fig, ax = plt.subplots()
    blue,red = pair
    itera = data[0]
    ax.plot(itera, data[blue], 'b', label = legend[blue])

    ax.plot(itera, data[red], 'r', label = legend[red])
    #ax.axis('legend[index]')
    plt.legend(loc='best')
    plt.xlabel('Trained_Iterations')
    print legend[red]
    if 'Loss' in legend[blue]:
        plt.ylabel('Loss')
    else:
        plt.ylabel('Accuracy')
    if 'MSE' in legend[red]:
        plt.ylabel('Loss(Blue) Reward(Red)')
    filename = "_".join(legend[pair])
    print filename
    plt.savefig(filename + '.png', dpi=600)
    print  log_filename
    #plt.show()



sys.exit()

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

save_fname = name + '.png'
plt.legend(loc='best')
plt.savefig("./pngs/" + save_fname)


