import numpy as np
import cv2,os,sys


path = sys.argv[1] + '/'
output_name = "merged.png"

h_l = 200
h_l = 0#200
h_r = 100
w_l = 300
w_r = 100
ratio = 0.3333
#ratio = 1

merged_img = None

fnames = sorted(os.listdir(path))

cnt = 0
for filename in fnames:
    if output_name in filename:
        continue
    if "Demo" in filename:
        continue
    if cnt == 9:
        break
    if ".png" in filename:
        print "Processing " + path + filename
        img = cv2.imread(path + filename)

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

        cnt += 1

#merged_img = cv2.resize(merged_img, (0,0), fx = ratio, fy = ratio, interpolation=cv2.INTER_CUBIC) # TODO

#cv2.imshow("merged", merged_img)

output_name = "_".join( (path.replace(".","") + output_name).split("/") )
print "Creating" + output_name
print
cv2.imwrite(output_name, merged_img)
cv2.waitKey()
