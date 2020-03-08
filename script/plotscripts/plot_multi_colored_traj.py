from process_scripts import visualize_multi_scripts
import numpy as np
import sys
from arg_parser import *
import matplotlib.pyplot as plt

fnames = sys.argv[1:]
scripts = []
for fname in fnames:
    trajf = open(fname)
    lines = trajf.readlines()[1:]
    script = np.array([np.array(map(lambda x: float(x), line.split())) for line in lines])

    output_path = "plot_traj_output/"
    fname_base = ""
    filename = output_path + "multi_traj_" + fname_base
    scripts.append(script)
visualize_multi_scripts(scripts, dist = "Demonstration #%d", filename = filename, denormal = False, write_traj = False)
