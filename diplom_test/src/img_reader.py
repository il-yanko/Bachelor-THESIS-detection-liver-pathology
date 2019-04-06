from os import listdir
import glob
import imageio
import numpy as np
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

class IMGReader:
    def __init__(self):
        pass

    @staticmethod
    def read_directory(dir_path, format=None):
        images_name = []
        images = []
        try:
            # images = [np.asarray(imageio.imread(img_path, as_gray=True), dtype=np.uint8)
            #           for img_path in glob.glob(dir_path + "*" + (("." + format) if format else ""))]
            images = [np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8)
                      for img_path in glob.glob(dir_path + "*" + (("." + format) if format else ""))]
            images_name = listdir(dir_path)
        except Exception as e:
            print(e)

        # images = map(np.ndarray.astype(np.uint8), images)
        return images_name, images
