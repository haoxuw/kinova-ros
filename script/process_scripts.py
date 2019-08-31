mini = True
mini = False
if mini == True:
    __OUTNAME__ = "mini"
    __AFFINE_MULTIPLIER__ = 50
    __MUTATE_MULTIPLIER__ = 10
else:
    __OUTNAME__ = "fake"
    __AFFINE_MULTIPLIER__ = 10
    __MUTATE_MULTIPLIER__ = 5000


import math

import sys,os,json,pickle

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm

import matplotlib.pyplot as plt
import numpy as np

def save_to_npy(data, name):
    print "saving to file " + name
    np.save(name, data)

def load_from_npy(name):
    print "loading to file " + name
    data = np.load(name)
    return data

def visualize_script(script):
    duration = script[-1][0]
    #t = np.array([(1 - t/duration, 0, t/duration) for t in script[:,0]])
    t = plt.cm.winter(np.array([t/duration for t in script[:,0]]))
    x = script[:,1]
    y = script[:,2]
    z = script[:,3]

    roll = script[:,4] - math.pi / 2
    pitch = script[:,5] 
    yaw = script[:,6]
    w = - np.array([math.sin(_v_) for _v_ in roll])
    xy = - np.array([math.cos(_v_) for _v_ in roll])
    u = - xy * np.array([math.sin(_v_) for _v_ in yaw])
    v = xy * np.array([math.cos(_v_) for _v_ in yaw])

    fig = plt.figure()
    fig.suptitle('Trajectory Visualization')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    ax = fig.gca(projection='3d')
    ax.view_init(elev=20, azim=100)

    img = ax.quiver(x, y, z, u, v, w, length=0.1, color = t, pivot = 'tip', normalize=True)

    ax.set_xlabel('<-left-  X axis  -right->')
    ax.set_xlim3d(-0.5,0.3)
    ax.set_ylabel('<-back-  Y axis  -front->')
    ax.set_ylim3d(-0.7,0)

    ax.set_zlabel('<-down-  Z axis  -up->')
    ax.set_zlim3d(0,0.5)

    m = cm.ScalarMappable(cmap=cm.winter)
    m.set_array(t*duration)
    cbar = plt.colorbar(m, extend = 'max', boundaries = range(0,int(duration)))
    cbar.set_label('elapsed_time')


    #cbar = fig.colorbar(img)
    plt.show()

    return script

def write_script_to_file(dist, script, fname):
    with open(fname,'w') as sfile:
        string = "# " + str(dist) + "\n"
        sfile.write(string)
        for line in script:
            string = " ".join([str(num) for num in line]) + "\n"
            sfile.write(string)
        print "writhing to %s : %d lines dist = %f" % (fname,len(script), dist)
    return
    

def write_string_to_file(string, fname):
    with open(fname,'w') as sfile:
        sfile.write(string)
        print "writhing to %s : %d chars" % (fname,len(string))
    return

def load_traj(filename):
    if filename[-5:] == '.traj':
        trajfile = open(filename)
        annotation = trajfile.readline()
        dist = float(annotation.split(" ")[1])
        string = trajfile.readlines()
        script = [line.split() for line in string]
        script = np.array(script).astype(float)
        return dist,script
    else:
        #print filename + " is not a traj file, skipped"
        return None, None

def load_task_file_under_path(path, n = -1):
    names = []
    dists = []
    scripts = []
    for i,filename in enumerate(sorted(os.listdir(path))):
        full_path = path+filename
        dist,script = load_traj(full_path)
        if script is None:
            continue
        names.append(filename)
        dists.append(dist)
        scripts.append(script)
        if (n != -1):
            if i+1 >= n:
                break
        #write_string_to_file(scri, full_path[:-5]+".traj")
    return names, np.array(dists).reshape(-1,1), np.array(scripts)

MAX_TIME = 32
mutator = {
    'z': [0, 0.1], #z
    't': [-20,20],    #theta
    's': [0, 3.0]     #start
}

def fix_point(t, hz):
    return float(int(float(t)*hz)) / hz

HZ = 2
def expand_script(scri):
    __EXPAND_RATE__ = 10
    expand_hz = __EXPAND_RATE__ * HZ

    start = np.copy(scri[0])
    start[0] = 0
    end = np.copy(scri[-1])
    if (end[0] > MAX_TIME):
        print "MAX_TIME smaller than script end time, sys exit"
        sys.exit(-4)
    end[0] = MAX_TIME
    scri = np.insert(scri, 0, start, axis=0)
    scri = np.insert(scri, -1, end, axis=0)


    long_scri = []
    current_t = 0
    for i in range(len(scri) - 1):
        start = np.array(scri[i])
        end = np.array(scri[i+1])
        sta_t = fix_point(start[0], expand_hz)
        end_t = fix_point(end[0], expand_hz)
        duration = int((end_t - sta_t) * expand_hz)
        if duration == 0:
            continue
        for i in range(duration + 1):
            between = start + ((end - start) / duration * i)
            if (between[0] < 1e-6):
                continue
            if int(fix_point(between[0], expand_hz) * expand_hz) > current_t:
                current_t += 1
                long_scri.append(between)
    long_scri = np.array(long_scri)
    filtered_scri = long_scri[0::__EXPAND_RATE__][:MAX_TIME*expand_hz/__EXPAND_RATE__]
    return filtered_scri

def rotate_2d(point,center,angle):
    px = point[0]
    py = point[1]
    cx = center[0]
    cy = center[1]

    px -= cx
    py -= cy
    x =   px * math.cos(angle) + py * math.sin(angle)
    y = - px * math.sin(angle) + py * math.cos(angle)
    x += cx
    y += cy
    return x,y

                      
def affine_script(scri):
    duration = scri[-1][0]

    scri = scri[:]
    z = np.random.uniform(*mutator['z'])
    scri[:,3] += z

    t = np.random.uniform(*mutator['t']) / 180 * math.pi
    for line in scri:
        x,y = rotate_2d (line[1:3], [0,0], t)
        line[1] = x
        line[2] = y
    scri[:,6] += t

    s = np.random.uniform(*mutator['s'])
    scri[:,0] += s

    return scri


def mutate_script(scri):
    MUTATION_NUM = MAX_TIME * HZ / 10
    MUTATION_ratio_MAX = 0.05

    dist = 0.0
    for i in range(MUTATION_NUM):
        ratio = np.random.uniform(-MUTATION_ratio_MAX, MUTATION_ratio_MAX)
        index = np.random.randint(scri.shape[0])
        column = np.random.randint(6) + 1
        dist += abs(ratio)
        scri[index,column] *= ratio + 1
    return dist, scri

def dump_np_array(names, dists, scripts, path, output_suffix, output_traj_files = False):

    cnt = 0
    arr_traj = []
    arr_dist = []
    for index, (name, dist, scri_ori) in enumerate(zip(names, dists, scripts)):
        print "processing " + name
        for i in range(__AFFINE_MULTIPLIER__):
            scri = np.copy(scri_ori)
            scri = affine_script(scri)
            #visualize_script(scri)
            scri = expand_script(scri[:])
            #visualize_script(scri)
            dist = 0
            if output_traj_files:
                write_script_to_file(dist, scri, output_path + "/fake_" + str(cnt) + "_affine.traj")
                cnt += 1

            arr_traj.append(scri)
            arr_dist.append(dist)

            scri_mut = np.copy(scri)
            for i in range(__MUTATE_MULTIPLIER__):
                # accumulative
                delta_dist, scri_mut = mutate_script(scri_mut)
                dist += delta_dist
                #visualize_script(scri_mut)
                if output_traj_files:
                    write_script_to_file(dist, scri_mut, output_path + "/fake_" + str(cnt) + "_mutate.traj")
                    cnt += 1
                arr_traj.append(scri)
                arr_dist.append(dist)
    #visualize_script(scri_mut)
    arr_traj = np.array(arr_traj)
    arr_dist = np.array(arr_dist)
    print arr_traj.shape
    print arr_dist.shape
    save_to_npy(arr_traj, path + "x_" + output_suffix)
    save_to_npy(arr_dist, path + "y_" + output_suffix)
    
def create_fake_trajs():
    names, dists, scripts = load_task_file_under_path(path);

    test = 2
    dump_np_array(names[:2], dists[:2], scripts[:2], output_path, "test.npy")
    dump_np_array(names[2:], dists[2:], scripts[2:], output_path, "train.npy")

def visualize_trajs(path):
    names, dists, scripts = load_task_file_under_path(path, 100);

    for name, dist, scri_ori in zip(names, dists, scripts):
        visualize_script(scri_ori)


path = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/tracked_results/"
output_path = "/home/haoxuw/mcgill/kinova/src/kinova-ros/script/" + __OUTNAME__ + "_results/"
def main():
    create_fake_trajs()

    #visualize_trajs(output_path)

if __name__== "__main__":
    main()
            


