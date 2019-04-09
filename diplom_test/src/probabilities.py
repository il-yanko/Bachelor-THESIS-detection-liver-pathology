from sklearn.preprocessing import normalize
from glcm import GLCM
from img_reader import IMGReader

import numpy as np
import pandas as pd

PROJECT_PWD = "/Users/ilyakachko/Diploma/diplom_test"

# PROJECT_PWD = "/Users/ikachko/Diploma/algorithm-detects-liver-pathology/diplom_test"

NORMA_DIR = PROJECT_PWD + "/norma/"
PATHOLOGY_DIR = PROJECT_PWD + "/pathology/"

norma_imgs_names, norma_imgs = IMGReader.read_directory(NORMA_DIR)
pathology_img_names, pathology_imgs = IMGReader.read_directory(PATHOLOGY_DIR)

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

dict_glcm_0 = {
    'name': [],
    'energy': [],
    'contrast': [],
    'homogenity': [],
    'correlation': [],
    'variance': [],
    'entropy': [],
    'pathology': []
}

for name, img in zip(pathology_img_names, pathology_imgs):
    dict_glcm_0['name'].append(name)
    glcm = GLCM(img)
    matrix = glcm.glcm_0()

    probs = count_probabilities_arr(matrix)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_0['energy'].append(glcm_eq.energy())
    dict_glcm_0['contrast'].append(glcm_eq.contrast())
    dict_glcm_0['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_0['correlation'].append(glcm_eq.correlation())
    dict_glcm_0['variance'].append(glcm_eq.variance())
    dict_glcm_0['entropy'].append(glcm_eq.entropy())
    dict_glcm_0['pathology'].append(1)

for name, img in zip(norma_imgs_names, norma_imgs):
    dict_glcm_0['name'].append(name)
    glcm = GLCM(img)
    matrix = glcm.glcm_0()

    probs = count_probabilities_arr(matrix)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_0['energy'].append(glcm_eq.energy())
    dict_glcm_0['contrast'].append(glcm_eq.contrast())
    dict_glcm_0['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_0['correlation'].append(glcm_eq.correlation())
    dict_glcm_0['variance'].append(glcm_eq.variance())
    dict_glcm_0['entropy'].append(glcm_eq.entropy())
    dict_glcm_0['pathology'].append(0)

df_glcm_0 = pd.DataFrame(dict_glcm_0)


print(df_glcm_0)
# g = glcm.glcm_complex()

# p = count_probabilities_arr(g)
# #
# gleq = GLCMEquations(p)
# print("Energy: ", gleq.energy())
# print("Contrast: ", gleq.contrast())
# print("Homogenity: ", gleq.homogenity())
# print("Correlation: ", gleq.correlation())
# print("Variance:", gleq.variance())
# print("Entropy: ", gleq.entropy())
