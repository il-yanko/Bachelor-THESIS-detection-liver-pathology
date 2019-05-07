import cv2
import glob

def png_to_bmp(full_dir):
    for file in glob.glob(full_dir + "*.png"):
        img = cv2.imread(file)
        cv2.imwrite(file[:-3] + "bmp", img)


def bmp_to_png(full_dir):
    for file in glob.glob(full_dir + "*.bmp"):
        img = cv2.imread(file)
        cv2.imwrite(file[:-3] + "png", img)