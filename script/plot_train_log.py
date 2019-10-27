import sys
import matplotlib.pyplot as plt
import numpy as np

with open(sys.argv[1]) as train_log:
    lines = train_log.readlines()[1:]
    data = np.array([[float(num) for num in line.split()] for line in lines]).transpose()
    itera, d_train_loss, d_train_accuracy, g_train_loss, g_train_MAE, d_eval_loss, d_eval_accuracy, g_eval_loss, g_eval_MAE, traj_eval_MSE = data
    legend = "itera, fit_loss_discriminator, d_train_accuracy, fit_loss_generator, g_train_MAE, eval_loss_discriminator, eval_accuracy_discriminator, eval_loss_generator, eval_accuracy_generator, traj_eval_MSE".replace(" ", "").split(",")


    print legend

    plotting = [[1,3], [5,7], [7,9], [6,8]]
    plotting = [[6,8]]
    for i in plotting:
        fig, ax = plt.subplots()
        blue,red = i
        ax.plot(itera, data[blue], 'b', label = legend[blue])

        ax.plot(itera, data[red], 'r', label = legend[red])
        #ax.axis('legend[index]')
        plt.legend(loc='best')
        plt.xlabel('training iterations')
        #plt.ylabel('Cross Entropy Loss')
        plt.ylabel('Accuracy')
        filename = str(i)
        plt.savefig(filename + '.png', dpi=600)

        plt.show()


