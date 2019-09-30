import cv2,os

h = 800
w = 800
x = 200
y = 100

fnames = sorted(os.listdir("."))
for filename in fnames:
    if ".jpg" in filename:
        with open(filename) as jpgf:
            print filename
            img = cv2.imread(filename)
            
            crop_img = img[y:y+h, x:x+w]
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(10)
