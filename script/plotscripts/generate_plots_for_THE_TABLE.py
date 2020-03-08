import sys, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

root_path = sys.argv[1]

BCST = "BC_State_FullTrajectory"
BCSS = "BC_State_State"
GST = "GAIL_State_FullTrajectory"
GSS = "GAIL_State_State"

EXPS = [BCST, BCSS, GST, GSS]

def parse_log(exp_log):
    exp_log = open(root_path + "/" + exp_log + ".txt").readlines()[1:]
    exp_log = np.array([[float(num) for num in line.split()] for line in exp_log])[:,[0,-2,-1]]
    return exp_log

data = {
    name: parse_log(name) for name in EXPS
}

data[BCST][:,0] *= 7475
data[BCSS][:,0] *= 7475


data[GST][:,0] *= 128
data[GSS][:,0] *= 128

for name,dataset in data.items():
    print name, " train ", np.array([row[1] for row in dataset if (row[0] > 6e5 and row[0] < 7e5)]).mean()
    print name, " eval ", np.array([row[2] for row in dataset if (row[0] > 6e5 and row[0] < 7e5)]).mean()
            

sys.exit()

for title in [[1,"_Train"],[2,"_Eval"]]:
    fig, ax = plt.subplots()
    for key, value in data.items():
        ax.plot(value[:,0], value[:,title[0]], label = key)

    plt.xlabel('Total_Num_Data_Trained')
    plt.ylabel('MinOfMeanSquareErrors' + title[1] + ' (NegativeReward)')
    ax.set_ylim([0,0.5])

    plt.legend(loc='best')
    plt.savefig("THE_PLOT"+ title[1] + '.png', dpi=600)
    #plt.show()

sys.exit()
