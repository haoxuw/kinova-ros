from arg_parser import *

mini = True
mini = False
if mini == True:
    __OUTNAME__ = "_mini"
    __AFFINE_MULTIPLIER__ = 50
    __MUTATE_MULTIPLIER__ = 10
    __MUTATE_LENGTH__ = 10
else:
    __OUTNAME__ = ""
    __AFFINE_MULTIPLIER__ = args.max_size / 10
    __MUTATE_MULTIPLIER__ = 1
    __MUTATE_LENGTH__ = 4

import math

fixed_range = [
    [-0.8, 0.5],  #x
    [-0.9, 0.1],    #y
    [0.15, 0.5],  #z
    [1.1, 2.8],   #pitch
    [0, 0.1],     #roll
    [-0.5, 1.5],  #yaw
]


import sys,os,json,pickle

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm

import matplotlib.pyplot as plt
import numpy as np

def dequantize(data, min_max, start = 0):
    shape = data.shape
    data = data.reshape([-1, shape[-1]])
    for i in range(len(min_max)):
        if i == 4:
            data[:, i+start] = 0
            continue
        data[: , i + start] *= min_max[i][1] - min_max[i][0]
        data[: , i + start] += min_max[i][0]
    return data.reshape(shape)

def quantize(data, min_max, start = 0):
    shape = data.shape
    data = data.reshape([-1, shape[-1]])
    for i in range(len(min_max)):
        if i == 4:
            data[:, i+start] = 0
            continue
        data[: , i + start] -= min_max[i][0]
        data[: , i + start] /= min_max[i][1] - min_max[i][0]
    return data.reshape(shape)

def save_to_npy(data, name):
    print "saving to file " + name + " of shape"
    print data.shape
    np.save(name, data)

def load_from_npy(name, shape = None):
    if not os.path.exists(name):
        return None
    print "loading from file " + name
    data = np.load(name).astype('float32')
    if shape:
        data = data.reshape(shape)
    return data

def add_time_axis(script):
    size = script.shape[0]
    time_series = np.arange(0,size).astype(float)/2
    time_series = time_series.reshape(-1, 1)
    script = np.concatenate([time_series, script] , axis = 1)
    return script

def visualize_script(script, dist = None, filename = None, dequant = True, write_traj = False):
    script = np.copy(script)

    if (script.shape[1] == 6):
        script = add_time_axis(script)

    if dequant:
        script = dequantize(script, fixed_range, start = 1)

    if write_traj:
        write_script_to_traj(dist, script, filename + '.traj')

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

    title = 'Trajectory'
    if dist is None:
        pass
    elif type(dist) == np.ndarray:
        title += ' -- (Score: %2.4f)' % dist[0]
    elif type(dist) == str:
        title += ' -- ' + dist
    else:
        title += ' -- (Score: %r)' % dist
    fig.suptitle(title)

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
    ax.set_zlim3d(0.1,0.6)

    m = cm.ScalarMappable(cmap=cm.winter)
    m.set_array(t*duration)
    cbar = plt.colorbar(m, extend = 'max', boundaries = range(0,int(duration)))
    cbar.set_label('elapsed_time')

    #cbar = fig.colorbar(img)
    if filename:
        plt.savefig(filename + '.png', dpi=600)
    else:
        plt.show()
    plt.close()

    return None

def write_script_to_traj(dist, script, fname):
    with open(fname,'w') as sfile:
        string = "# " + str(dist) + "\n"
        sfile.write(string)
        for line in script:
            string = " ".join([str(num) for num in line]) + "\n"
            sfile.write(string)
        #print "writhing to %s : %d lines dist = %r" % (fname,len(script), dist)
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
affine = {
    'x': [-0.2, 0.2], 
    'y': [-0.2, 0.2], 
    'z': [-0.01, 0.01], #z
    'e': [-1,1],    #eps
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
    x = px * math.cos(angle) - py * math.sin(angle)
    y = px * math.sin(angle) + py * math.cos(angle)
    x += cx
    y += cy
    return x,y

                      
def affine_script(scri):
    duration = scri[-1][0]

    scri = scri[:]
    x = np.random.uniform(*affine['x'])
    scri[:,1] += x

    y = np.random.uniform(*affine['y'])
    scri[:,2] += y

    z = np.random.uniform(*affine['z'])
    scri[:,3] += z

    e = np.random.uniform(*affine['e']) / 180 * math.pi
    scri[:,4] += e

    scri[:,5] = 0

    t = np.random.uniform(*affine['t']) / 180 * math.pi
    for line in scri:
        x,y = rotate_2d (line[1:3], [0,0], t)
        line[1] = x
        line[2] = y
    scri[:,6] += t

    s = np.random.uniform(*affine['s'])
    scri[:,0] += s

    return scri


def mutate_script(scri):
    MUTATION_NUM = MAX_TIME * HZ / __MUTATE_LENGTH__
    MUTATION_ratio_MAX = 1 / __MUTATE_LENGTH__

    dist = 0.0
    for j in range(__MUTATE_LENGTH__):
        # accumulative
        for i in range(MUTATION_NUM):
            #ratio = np.random.uniform(-MUTATION_ratio_MAX, MUTATION_ratio_MAX)
            index = np.random.randint(scri.shape[0] - 1 ) + 1
            column = np.random.randint(6)
            if column == 4:
                continue
            assert(scri.shape[-1] == 7)
            ori = scri[index][column+1]
            new = np.random.uniform(fixed_range[column][0], fixed_range[column][1])
            #scri[index][column+1] = ori + (new - ori) * MUTATION_ratio_MAX
            scri[index][column+1] = new

            dist += 1

    #visualize_script(scri, dequant = False)
    return dist, scri

def dump_np_array(names, dists, scripts, output_path, output_suffix, plot_samples = None, output_traj_files = False):

    cnt = 0
    arr_traj = []
    arr_dist = []

    affine_arr_traj = []

    sample_img_dir = "/sample/"

    for index, (name, dist, scri_ori) in enumerate(zip(names, dists, scripts)):
        for i in range(__AFFINE_MULTIPLIER__):
            dist = 1

            filename = None
            scri = np.copy(scri_ori)
            #filename = output_path + "%02d_ori" %index
            #visualize_script(scri, filename, dist = "original", dequant = False)
            scri = expand_script(scri[:])
            #filename = output_path + "%02d_exp" %index
            #visualize_script(scri, filename, dist = "filled", dequant = False)

            scri = affine_script(scri)
            #filename = output_path + "%02d_rot" %index
            #visualize_script(scri, filename, dist = "2d_rotated", dequant = False)
            #visualize_script(scri)
            if output_traj_files:
                write_script_to_traj(dist, scri, output_path + "/fake_" + str(cnt) + "_affine.traj")
                cnt += 1

            arr_traj.append(np.copy(scri))
            arr_dist.append(np.copy(dist))

            affine_arr_traj.append(np.copy(scri))

            for j in range(__MUTATE_MULTIPLIER__):
                scri_mut = np.copy(scri)
                dist_mut = dist

                delta_dist, scri_mut = mutate_script(scri_mut)
                #visualize_script(scri_mut, dist = "scrambled", filename = output_path + "%02d_scr" %index, dequant = False)
                dist_mut += delta_dist

                if output_traj_files:
                    write_script_to_traj(dist, scri_mut, output_path + "/fake_" + str(cnt) + "_mutate.traj")
                    cnt += 1

                # now disc predict binary
                dist_mut = 0
                arr_traj.append(np.copy(scri_mut))
                arr_dist.append(np.copy(dist_mut))
                #visualize_script(scri_mut, dist = dist_mut, dequant = False)

    arr_traj = np.array(arr_traj)
    arr_dist = np.array(arr_dist)
    #arr_dist /= np.max(arr_dist)

    arr_traj = quantize(arr_traj, fixed_range, start = 1)

    save_to_npy(arr_traj, output_path + "x_" + output_suffix)
    save_to_npy(arr_dist, output_path + "y_" + output_suffix)
    

    affine_arr_traj = np.array(affine_arr_traj)
    affine_arr_traj = quantize(affine_arr_traj, fixed_range, start = 1)
    save_to_npy(affine_arr_traj, output_path + "affine_" + output_suffix)


    #after saved
    print "after quantize"
    print arr_traj.shape
    print "min:"
    print arr_traj.min(axis = 0).min(axis = 0)
    print "max:"
    print arr_traj.max(axis = 0).max(axis = 0)

    if plot_samples:
        os.system('mkdir -p ' + output_path + sample_img_dir)
        for i in range(0, len(affine_arr_traj), len(affine_arr_traj) / 5):
            filename = output_path + sample_img_dir + "/" + plot_samples + '_affile_sample_%d.png' %i
            traj = affine_arr_traj[i]
            #visualize_script(traj, dist = "affine")
            visualize_script(traj, filename, dist = "affine transformed")
        for i in range(0, len(arr_traj), len(arr_traj) / 5):
            filename = output_path + sample_img_dir + "/" + plot_samples + '_sample_%d.png' %i
            traj = arr_traj[i]
            #visualize_script(traj, dist = arr_dist[i])
            visualize_script(traj, filename, dist = "scrambled score %d" % arr_dist[i])
                
    arr_traj = dequantize(arr_traj, fixed_range, start = 1)
    print "after dequantize"
    print arr_traj.shape
    print "min:"
    print arr_traj.min(axis = 0).min(axis = 0)
    print "max:"
    print arr_traj.max(axis = 0).max(axis = 0)

def create_fake_trajs(input_path, output_path):
    if not output_path:
        return
    os.system('mkdir -p ' + output_path)
    
    names, dists, scripts = load_task_file_under_path(input_path);

    test = 2
    dump_np_array(names[:-2], dists[:-2], scripts[:-2], output_path, output_suffix = "train.npy")#, plot_samples = "train_data")
    dump_np_array(names[-2:], dists[-2:], scripts[-2:], output_path, output_suffix = "test.npy")#, plot_samples = "test_data")


def main():
    create_fake_trajs(args.input_dir, args.output_dir[:-1] + __OUTNAME__ + "/")

    #visualize_trajs(output_path)

if __name__== "__main__":
    main()
            


