from process_scripts import visualize_script
import numpy as np
import sys
from arg_parser import *
import matplotlib.pyplot as plt

fname = args.load_traj_file

with open(fname) as trajf:
    lines = trajf.readlines()[1:]
    script = np.array([np.array(map(lambda x: float(x), line.split())) for line in lines])
    print script.shape

    output_path = "plot_traj_output/"
    filename = None
    fname_base = fname.split('/')[-1].split('.')[-2]
    filename = output_path + "multi_view_" + fname_base
    num = int("".join([ch for ch in fname_base if ch.isdigit()]))
    #print len(script.shape)
    visualize_script(script, dist = "Multiple Views of Demonstration #%d" % num, filename = filename, denormal = False, write_traj = False, color = "reds", m = 2 , n = 2)
