import glob
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import os.path


class IMGReader:
    def __init__(self):
        pass

    @staticmethod
    def read_directory(dir_path, file_format=None):
        try:
            images = [np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8)
                      for img_path in glob.glob(dir_path + "*" + (("." + file_format) if file_format else ""))]
            print("It was loaded", len(images), "images from", dir_path)
            return images
        except Exception as e:
            print(e)
            return


# ALTERNATIVE LOADER:
# process RGB/grayscale
def rgb_to_gray(rgb):
    # scalar product of colors with certain theoretical coefficients according to the YUV system
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).round(3).astype(int)


# download folder BMP
def get_all_bmp(full_dir):
    # to calculate number of files in the folder
    file_number = len(next(os.walk(full_dir))[2])
    # print(fileNumber, "files were found")
    img_arr = list()
    for i in range(1, file_number + 1):
        img_arr.append(mpimg.imread(full_dir + '/' + str(i) + ".bmp"))
    print(len(img_arr), "images were downloaded")
    return img_arr


def get_all_img_make_gray(cwd, folder_name):
    path = cwd + "/" + folder_name
    print("\nPath = ", path)
    img_arr = get_all_bmp(path)
    for i in range(len(img_arr)):
        img_arr[i] = rgb_to_gray(img_arr[i])
    return img_arr
