from process_scripts import visualize_multi_scripts
from process_scripts import visualize_script
import numpy as np
import sys
from arg_parser import *
import matplotlib.pyplot as plt

fnames = sys.argv[1:]
scripts = []
for i, fname in enumerate(fnames):
    trajf = open(fname)
    lines = trajf.readlines()[1:]
    script = np.array([np.array(map(lambda x: float(x), line.split())) for line in lines])

    output_path = "plot_traj_output/"
    fname_base = fname.split(".")[-2].split("/")[-1]
    filename = output_path + "demo_traj_" + fname_base
    #scripts.append(script)
    visualize_script(script, dist = "Demonstration #%d" %i, filename = filename, denormal = False, write_traj = False, color = 'reds')
#visualize_multi_scripts(scripts, dist = "Multiple Demonstrations", filename = filename, denormal = False, write_traj = False)
