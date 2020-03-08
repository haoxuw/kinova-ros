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


critera = [[1,3],[2,4],[5,7],[6,8],[2,9],[4,9],[6,10],[8,10],]

for pair in critera:
    fig, ax = plt.subplots()
    blue,red = pair
    itera = data[0]
    #ax.axis('legend[index]')
    plt.legend(loc='best')
    plt.xlabel('Trained_Iterations')
    print legend[red]
    if 'Loss' in legend[blue]:
        plt.ylabel('Loss')
    else:
        plt.ylabel('Accuracy')


    ax.plot(itera, data[blue], 'b', label = legend[blue])
    if 'MSE' in legend[red]:
        plt.ylabel('Accuracy(Blue) NegativeReward(Orange)')
        ax.plot(itera, data[red], 'orange', label = legend[red])
    else:
        ax.plot(itera, data[red], 'r', label = legend[red])

    filename = "_".join(legend[pair])
    print filename
    print log_filename
    plt.legend(loc='best')
    plt.savefig(filename + '.png', dpi=600)
    #plt.show()

