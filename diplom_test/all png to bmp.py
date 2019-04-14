import cv2
import glob

for file in glob.glob("*.png"):
    img = cv2.imread(file)
    cv2.imwrite(file[:-3] + "bmp", img)
