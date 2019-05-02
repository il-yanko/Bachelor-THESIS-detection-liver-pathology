import glob
import numpy as np
from PIL import Image

class IMGReader:
    def __init__(self):
        pass
    @staticmethod
    def read_directory(dir_path, format=None):
        images = []
        try:
            images = [np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8)
                      for img_path in glob.glob(dir_path + "*" + (("." + format) if format else ""))]
            print(len(images), "фото было загружено")
        except Exception as e:
            print(e)
        return images


# ALTERNATIVE LOADER:
'''
import os.path
# process RGB/grayscale
def rgb_to_gray(rgb):
    #скалярное произведение начальных цветов с определенными теоретическими коэффициентами по системе YUV
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).round(3).astype(int)

# download folder BMP
def get_all_BMP(fullDir):
    # to calculate number of files in the folder
    fileNumber = len(next(os.walk(fullDir))[2])
    # print(fileNumber, "files were found")
    imgArr = []
    for i in range(1, fileNumber + 1):
        imgArr.append(mpimg.imread(fullDir + '/' + str(i) + ".bmp"))
    print(len(imgArr), "images were downloaded")
    return imgArr

def get_all_img_make_gray(cwd, folderName):
    path = cwd + "/" + folderName
    print("\nPath = ", path)
    imgArray = get_all_BMP(path)
    for i in range(len(imgArray)):
        imgArray[i] = rgb_to_gray(imgArray[i])
    return imgArray
'''
