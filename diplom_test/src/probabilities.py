from sklearn.preprocessing import normalize
from glcm import GLCM
from img_reader import IMGReader

import numpy as np

PROJECT_PWD = "/Users/ikachko/Diploma/algorithm-detects-liver-pathology/diplom_test"

NORMA_DIR = PROJECT_PWD + "/norma/"
PATHOLOGY_DIR = PROJECT_PWD + "/pathology/"

img_names, images = IMGReader.read_directory(NORMA_DIR)


def count_probabilities_arr(array):
    print(array)
    vector = np.asarray(array.flatten(), dtype=np.float64)
    sum_of_all = vector.sum()

    prob_matrix = np.zeros(array.shape)

    for i in range(len(array)):
        for j in range(len(array[0])):
            prob_matrix[i][j] = array[i][j]/sum_of_all
    return prob_matrix

class GLCMEquations:
    def __init__(self, matrix):
        self.matrix = matrix

    # f1
    def energy(self):
        return np.sum(np.power(self.matrix.flatten(), 2))

    # f2
    def contrast(self):
        res_contrast = 0

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                res_contrast += self.matrix[i][j] * (i - j) ** 2
        return res_contrast

    def homogenity(self):
        res_homog = 0

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                res_homog += self.matrix[i][j]/(1 + (i - j) ** 2)

        return res_homog

    # f3
    def correlation(self):
        variance = self.variance()
        mean = self.mean()

        res_corr = 0
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                res_corr += self.matrix[i][j]\
                           * (((i - mean) * (j - mean)) / variance**2)
        return res_corr

    def mean(self):
        return self.matrix.flatten().mean()
    # f4
    def variance(self):
        mean = self.mean()

        result = 0
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                result += (i - mean) ** 2 * self.matrix[i][j]
        return result
    # f5
    def inverse_difference_moment(self):
        pass

    # f6
    def sum_average(self):
        pass

    # f7
    def sum_variance(self):
        pass

    # f8
    def sum_entropy(self):
        pass

    # f9
    def entropy(self):
        res_entr = 0
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i][j] != 0:
                    res_entr += -np.log(self.matrix[i][j]) * self.matrix[i][j]
        return res_entr

    # f10
    def differnce_variance(self):
        pass

    # f11
    def differnce_entropy(self):
        pass

    # f14
    def max_correlation_coeff(self):
        pass

glcm = GLCM(images[0])

g = glcm.glcm_complex()

p = count_probabilities_arr(g)
#
gleq = GLCMEquations(p)
print("Energy: ", gleq.energy())
print("Contrast: ", gleq.contrast())
print("Homogenity: ", gleq.homogenity())
print("Correlation: ", gleq.correlation())
print("Variance:", gleq.variance())
print("Entropy: ", gleq.entropy())
