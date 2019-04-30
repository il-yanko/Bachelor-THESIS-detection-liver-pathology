import numpy as np


class GLRLM:
    """
    GLRLM - Gray level run length matrix
    """

    def __init__(self, image):
        self.image = image
        self.h, self.w = np.shape(image)
        self.gray_level = 255
        self.glrlm = None


    def glrlm_0(self):
        """Vertical run"""

        glrlm = np.zeros([self.gray_level, self.w], dtype=int)

        N_z = 0

        for i in range(self.h):
            count = 0
            for j in range(self.w):
                if j < self.w - 1 and self.image[i][j] == self.image[i][j + 1]:
                    count += 1
                else:
                    glrlm[self.image[i][j]][count] += 1
                    N_z += 1
                    count = 0
        return glrlm, N_z

class GLRLM_Features():

    @staticmethod
    def SRE(matrix, N_z):
        """
        Short Run Emphasis

        N_z - number of runs in the image along angle
        """
        if matrix is None:
            raise AttributeError("No matrix")

        res = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                res += matrix[i][j] * j * j

        res /= N_z

        return res

    @staticmethod
    def LRE(matrix, N_z):
        """Long Run Emphasis"""

        if matrix is None:
            raise AttributeError("No matrix")

        res = 0
        N_g, N_r = matrix.shape[0], matrix.shape[1]

        for i in range(N_g):
            for j in range(N_r):
                res += matrix[i][j] / (j + 1 ** 2)

        res /= N_z

        return res



    @staticmethod
    def LGRE(matrix, N_z):
        """Low Gray Level Run Rmphasis"""
        if matrix is None:
            raise AttributeError("No matrix")

        res = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                res += (matrix[i][j]) / ((i + 1) ** 2)
        res /= N_z

        return res

    @staticmethod
    def HGRE(matrix, N_z):
        """High Rray Level Run Emphasis"""
        if matrix is None:
            raise AttributeError("No matrix")

        res = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                res += (matrix[i][j] * (i * i))

        res /= N_z

        return res

    @staticmethod
    def GLNU(matrix, N_z):
        """Gray Level Non-uniformity"""
        if matrix is None:
            raise AttributeError("No matrix")

        res = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                res += matrix[i][j] ** 2
        res /= N_z

        return res

from img_reader import IMGReader
import pandas as pd

PROJECT_PWD = "/Volumes/Storage/goinfre/ikachko/algorithm-detects-liver-pathology"
NORMA_DIR = PROJECT_PWD + "/norma_png/"
PATHOLOGY_DIR = PROJECT_PWD + "/pathology_png/"

norma_imgs_names, norma_imgs = IMGReader.read_directory(NORMA_DIR)
pathology_img_names, pathology_imgs = IMGReader.read_directory(PATHOLOGY_DIR)


features = ['SRE', 'LRE', 'LGRE', 'HGRE', 'GLNU', 'isPatho']

data = []

for img_name, img in zip(norma_imgs_names, norma_imgs):
    print(img_name)
    g = GLRLM(img)

    m, N_z = g.glrlm_0()

    d = {
        # 'SRE': GLRLM_Features().SRE(m, N_z),
        # 'LRE': GLRLM_Features().LRE(m, N_z),
        'LGRE': GLRLM_Features().LGRE(m, N_z),
        'HGRE': GLRLM_Features().HGRE(m, N_z),
        'GLNU': GLRLM_Features().GLNU(m, N_z),
        'isPatho': 0
    }
    data.append(d)

for img in pathology_imgs:
    g = GLRLM(img)

    m, N_z = g.glrlm_0()

    d = {
        # 'SRE': GLRLM_Features().SRE(m, N_z),
        # 'LRE': GLRLM_Features().LRE(m, N_z),
        'LGRE': GLRLM_Features().LGRE(m, N_z),
        'HGRE': GLRLM_Features().HGRE(m, N_z),
        'GLNU': GLRLM_Features().GLNU(m, N_z),
        'isPatho': 1
    }
    data.append(d)

df = pd.DataFrame(data).sample(frac=1)
df.to_csv('./datasets/glrlm_0.csv')
print(df.head())

