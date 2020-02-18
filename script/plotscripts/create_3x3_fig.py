import numpy as np
import cv2,os,sys

assert len(sys.argv) == 3

input_path = sys.argv[1] + '/'
output_name = sys.argv[2]

h_l = 0
h_r = 0
w_l = 300
w_r = 100
ratio = 0.3333
#ratio = 1
output_name_prefix = "_3x3_"


def plot_9(fnames, name = "test_plot", path = "."):
    assert len(fnames) == 9
    merged_img = None
    for cnt, filename in enumerate(fnames):
        img = cv2.imread(filename)
    
        h,w,c = img.shape
        img = img[h_l : h - h_r, w_l : w - w_r, :]
        img = cv2.resize(img, (0,0), fx = ratio, fy = ratio, interpolation=cv2.INTER_CUBIC)
    
        h,w,c = img.shape
    
        if merged_img is None:
            merged_img = np.ones([h*3, w*3, c]).astype(img.dtype) * 255
        x = cnt % 3
        y = cnt / 3
        hs = y * h
        ws = x * w
        merged_img[hs : hs + h , ws : ws + w , : ] = img

    output_name = path + "/" + output_name_prefix + name + ".png"
    print "Creating" + output_name
    print
    cv2.imwrite(output_name, merged_img)
    #cv2.imshow("merged", merged_img)
    #cv2.waitKey()
'''
fname_list = sorted(os.listdir(path))
logf_path = path + "/train_log.txt"
if not os.path.isfile(logf_path):
    print logf_path , "log file not found"
    sys.exit()
sys.exit()
'''

mid_string = "_Test_Predicted_In_Iteration_"
mid_string = "_Test_Predicted_After_Epoch_"
mid_string = "_Train_Predicted_After_Epoch_"
fname_list = sorted(os.listdir(input_path))
fnames = {}
epis = set()
for fname in fname_list:
    if output_name_prefix in fname:
        continue
    if mid_string not in fname:
        continue
    if '.png' not in fname:
        continue
    if "Demo" in fname:
        continue
    episo, itera= map(lambda x: int(x), fname.replace("#","").replace(".png","").split(mid_string))
    epis.add(episo)
    fnames[episo*1000000 + itera] = fname
fnames = np.array([ fnames[k] for k in sorted(fnames.keys()) ]).reshape(len(epis), -1)

print "collected: ", fnames.shape

outpath = './pngoutput_%s' % output_name
os.system('mkdir %s' % outpath)
for i in range(fnames.shape[0]):
    picked_fnames = fnames[i,::fnames.shape[1] / (9)][:9]
    print picked_fnames.shape
    if not i:
        print "picked " + "  ".join([name[-10:] for name in picked_fnames])

    picked_fnames = map(lambda x : input_path + x, picked_fnames) 
    plot_9(picked_fnames, "_%d_%s_" % (i,output_name), path = outpath)

