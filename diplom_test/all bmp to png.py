import cv2
import glob

for file in glob.glob("*.bmp"):
    img = cv2.imread(file)
    cv2.imwrite(file[:-3] + "png", img)
