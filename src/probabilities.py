from sklearn.preprocessing import normalize
from glcm import GLCM
from img_reader import IMGReader

import numpy as np
import pandas as pd

PROJECT_PWD = "/Users/ilyakachko/Diploma/algorithm-detects-liver-pathology/diplom_test"

# PROJECT_PWD = "/Users/ikachko/Diploma/algorithm-detects-liver-pathology/diplom_test"

NORMA_DIR = PROJECT_PWD + "/norma/"
PATHOLOGY_DIR = PROJECT_PWD + "/pathology/"

norma_imgs_names, norma_imgs = IMGReader.read_directory(NORMA_DIR)
pathology_img_names, pathology_imgs = IMGReader.read_directory(PATHOLOGY_DIR)

def count_probabilities_arr(array):
    vector = np.asarray(array.flatten(), dtype=np.float64)
    # print(vector)

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

    def glcm_mean(self):
        m = 0
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                m += i * self.matrix[i][j]
        return

    def glcm_variance(self):
        v = 0
        m = self.glcm_mean()
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                v += self.matrix[i][j] * ((i - m) ** 2)
        return v

    def calc_A(self):
        pass

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

dict_glcm_45 = {
    'name': [],
    'energy': [],
    'contrast': [],
    'homogenity': [],
    'correlation': [],
    'variance': [],
    'entropy': [],
    'pathology': []
}


dict_glcm_90 = {
    'name': [],
    'energy': [],
    'contrast': [],
    'homogenity': [],
    'correlation': [],
    'variance': [],
    'entropy': [],
    'pathology': []
}


dict_glcm_135 = {
    'name': [],
    'energy': [],
    'contrast': [],
    'homogenity': [],
    'correlation': [],
    'variance': [],
    'entropy': [],
    'pathology': []
}


dict_glcm_complex = {
    'name': [],
    'energy': [],
    'contrast': [],
    'homogenity': [],
    'correlation': [],
    'variance': [],
    'entropy': [],
    'pathology': []
}


dict_glcm_gen_duplex = {
    'name': [],
    'energy': [],
    'contrast': [],
    'homogenity': [],
    'correlation': [],
    'variance': [],
    'entropy': [],
    'pathology': []
}


dict_glcm_complex_duplex = {
    'name': [],
    'energy': [],
    'contrast': [],
    'homogenity': [],
    'correlation': [],
    'variance': [],
    'entropy': [],
    'pathology': []
}
# glcm = GLCM(pathology_imgs[0])
# g = glcm.glcm_0()
# p = count_probabilities_arr(g)
# gleq = GLCMEquations(p)
# print("Energy: ", gleq.energy())
# print("Contrast: ", gleq.contrast())
# print("Homogenity: ", gleq.homogenity())
# print("Correlation: ", gleq.correlation())
# print("Variance:", gleq.variance())
# print("Entropy: ", gleq.entropy())

for i, name, img in enumerate(zip(pathology_img_names, pathology_imgs)):
    glcm = GLCM(img)
    matrix_gl_0 = glcm.glcm_0()

    probs = count_probabilities_arr(matrix_gl_0)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_0['name'].append('patho_' + str(i))
    dict_glcm_0['energy'].append(glcm_eq.energy())
    dict_glcm_0['contrast'].append(glcm_eq.contrast())
    dict_glcm_0['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_0['correlation'].append(glcm_eq.correlation())
    dict_glcm_0['variance'].append(glcm_eq.variance())
    dict_glcm_0['entropy'].append(glcm_eq.entropy())
    dict_glcm_0['pathology'].append(1)

    matrix_gl_45 = glcm.glcm_45()
    probs = count_probabilities_arr(matrix_gl_45)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_45['name'].append('patho_' + str(i))
    dict_glcm_45['energy'].append(glcm_eq.energy())
    dict_glcm_45['contrast'].append(glcm_eq.contrast())
    dict_glcm_45['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_45['correlation'].append(glcm_eq.correlation())
    dict_glcm_45['variance'].append(glcm_eq.variance())
    dict_glcm_45['entropy'].append(glcm_eq.entropy())
    dict_glcm_45['pathology'].append(1)

    matrix_gl_90 = glcm.glcm_90()
    probs = count_probabilities_arr(matrix_gl_90)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_90['name'].append('patho_' + str(i))
    dict_glcm_90['energy'].append(glcm_eq.energy())
    dict_glcm_90['contrast'].append(glcm_eq.contrast())
    dict_glcm_90['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_90['correlation'].append(glcm_eq.correlation())
    dict_glcm_90['variance'].append(glcm_eq.variance())
    dict_glcm_90['entropy'].append(glcm_eq.entropy())
    dict_glcm_90['pathology'].append(1)

    matrix_gl_135 = glcm.glcm_135()
    probs = count_probabilities_arr(matrix_gl_135)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_135['name'].append('patho_' + str(i))
    dict_glcm_135['energy'].append(glcm_eq.energy())
    dict_glcm_135['contrast'].append(glcm_eq.contrast())
    dict_glcm_135['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_135['correlation'].append(glcm_eq.correlation())
    dict_glcm_135['variance'].append(glcm_eq.variance())
    dict_glcm_135['entropy'].append(glcm_eq.entropy())
    dict_glcm_135['pathology'].append(1)

    # matrix_gl_complex = glcm.glcm_complex()
    # probs = count_probabilities_arr(matrix_gl_complex)
    # glcm_eq = GLCMEquations(probs)
    # dict_glcm_complex['name'].append('patho_' + name)
    # dict_glcm_complex['energy'].append(glcm_eq.energy())
    # dict_glcm_complex['contrast'].append(glcm_eq.contrast())
    # dict_glcm_complex['homogenity'].append(glcm_eq.homogenity())
    # dict_glcm_complex['correlation'].append(glcm_eq.correlation())
    # dict_glcm_complex['variance'].append(glcm_eq.variance())
    # dict_glcm_complex['entropy'].append(glcm_eq.entropy())
    # dict_glcm_complex['pathology'].append(1)

    # matrix_gl_gen_duplex = glcm.glcm_gen_duplex()
    # probs = count_probabilities_arr(matrix_gl_gen_duplex)
    # glcm_eq = GLCMEquations(probs)
    # dict_glcm_gen_duplex['name'].append('patho_' + name)
    # dict_glcm_gen_duplex['energy'].append(glcm_eq.energy())
    # dict_glcm_gen_duplex['contrast'].append(glcm_eq.contrast())
    # dict_glcm_gen_duplex['homogenity'].append(glcm_eq.homogenity())
    # dict_glcm_gen_duplex['correlation'].append(glcm_eq.correlation())
    # dict_glcm_gen_duplex['variance'].append(glcm_eq.variance())
    # dict_glcm_gen_duplex['entropy'].append(glcm_eq.entropy())
    # dict_glcm_gen_duplex['pathology'].append(1)

    # matrix_gl_duplex_complex = glcm.glcm_complex_duplex()
    # probs = count_probabilities_arr(matrix_gl_duplex_complex)
    # glcm_eq = GLCMEquations(probs)
    # dict_glcm_complex_duplex['name'].append('patho_' + name)
    # dict_glcm_complex_duplex['energy'].append(glcm_eq.energy())
    # dict_glcm_complex_duplex['contrast'].append(glcm_eq.contrast())
    # dict_glcm_complex_duplex['homogenity'].append(glcm_eq.homogenity())
    # dict_glcm_complex_duplex['correlation'].append(glcm_eq.correlation())
    # dict_glcm_complex_duplex['variance'].append(glcm_eq.variance())
    # dict_glcm_complex_duplex['entropy'].append(glcm_eq.entropy())
    # dict_glcm_complex_duplex['pathology'].append(1)

    print("Pathology image " + name + " done.")
    print(str(len(pathology_img_names) - pathology_img_names.index(name)) + " pathology images left.")

for i, name, img in enumerate(zip(norma_imgs_names, norma_imgs)):
    glcm = GLCM(img)
    matrix_gl_0 = glcm.glcm_0()

    probs = count_probabilities_arr(matrix_gl_0)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_0['name'].append('norma_' + str(i))
    dict_glcm_0['energy'].append(glcm_eq.energy())
    dict_glcm_0['contrast'].append(glcm_eq.contrast())
    dict_glcm_0['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_0['correlation'].append(glcm_eq.correlation())
    dict_glcm_0['variance'].append(glcm_eq.variance())
    dict_glcm_0['entropy'].append(glcm_eq.entropy())
    dict_glcm_0['pathology'].append(0)

    matrix_gl_45 = glcm.glcm_45()
    probs = count_probabilities_arr(matrix_gl_45)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_45['name'].append('norma_' + str(i))
    dict_glcm_45['energy'].append(glcm_eq.energy())
    dict_glcm_45['contrast'].append(glcm_eq.contrast())
    dict_glcm_45['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_45['correlation'].append(glcm_eq.correlation())
    dict_glcm_45['variance'].append(glcm_eq.variance())
    dict_glcm_45['entropy'].append(glcm_eq.entropy())
    dict_glcm_45['pathology'].append(0)

    matrix_gl_90 = glcm.glcm_90()
    probs = count_probabilities_arr(matrix_gl_90)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_90['name'].append('norma_' + str(i))
    dict_glcm_90['energy'].append(glcm_eq.energy())
    dict_glcm_90['contrast'].append(glcm_eq.contrast())
    dict_glcm_90['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_90['correlation'].append(glcm_eq.correlation())
    dict_glcm_90['variance'].append(glcm_eq.variance())
    dict_glcm_90['entropy'].append(glcm_eq.entropy())
    dict_glcm_90['pathology'].append(0)

    matrix_gl_135 = glcm.glcm_135()
    probs = count_probabilities_arr(matrix_gl_135)
    glcm_eq = GLCMEquations(probs)
    dict_glcm_135['name'].append('norma_' + str(i))
    dict_glcm_135['energy'].append(glcm_eq.energy())
    dict_glcm_135['contrast'].append(glcm_eq.contrast())
    dict_glcm_135['homogenity'].append(glcm_eq.homogenity())
    dict_glcm_135['correlation'].append(glcm_eq.correlation())
    dict_glcm_135['variance'].append(glcm_eq.variance())
    dict_glcm_135['entropy'].append(glcm_eq.entropy())
    dict_glcm_135['pathology'].append(0)

    # matrix_gl_complex = glcm.glcm_complex()
    # probs = count_probabilities_arr(matrix_gl_complex)
    # glcm_eq = GLCMEquations(probs)
    # dict_glcm_complex['name'].append('norma_' + name)
    # dict_glcm_complex['energy'].append(glcm_eq.energy())
    # dict_glcm_complex['contrast'].append(glcm_eq.contrast())
    # dict_glcm_complex['homogenity'].append(glcm_eq.homogenity())
    # dict_glcm_complex['correlation'].append(glcm_eq.correlation())
    # dict_glcm_complex['variance'].append(glcm_eq.variance())
    # dict_glcm_complex['entropy'].append(glcm_eq.entropy())
    # dict_glcm_complex['pathology'].append(0)
    #
    # # matrix_gl_gen_duplex = glcm.glcm_gen_duplex()
    # # probs = count_probabilities_arr(matrix_gl_gen_duplex)
    # # glcm_eq = GLCMEquations(probs)
    # # dict_glcm_gen_duplex['name'].append('norma_' + name)
    # # dict_glcm_gen_duplex['energy'].append(glcm_eq.energy())
    # # dict_glcm_gen_duplex['contrast'].append(glcm_eq.contrast())
    # # dict_glcm_gen_duplex['homogenity'].append(glcm_eq.homogenity())
    # # dict_glcm_gen_duplex['correlation'].append(glcm_eq.correlation())
    # # dict_glcm_gen_duplex['variance'].append(glcm_eq.variance())
    # # dict_glcm_gen_duplex['entropy'].append(glcm_eq.entropy())
    # # dict_glcm_gen_duplex['pathology'].append(0)
    #
    # matrix_gl_duplex_complex = glcm.glcm_complex_duplex()
    # probs = count_probabilities_arr(matrix_gl_duplex_complex)
    # glcm_eq = GLCMEquations(probs)
    # dict_glcm_complex_duplex['name'].append('norma_' + name)
    # dict_glcm_complex_duplex['energy'].append(glcm_eq.energy())
    # dict_glcm_complex_duplex['contrast'].append(glcm_eq.contrast())
    # dict_glcm_complex_duplex['homogenity'].append(glcm_eq.homogenity())
    # dict_glcm_complex_duplex['correlation'].append(glcm_eq.correlation())
    # dict_glcm_complex_duplex['variance'].append(glcm_eq.variance())
    # dict_glcm_complex_duplex['entropy'].append(glcm_eq.entropy())
    # dict_glcm_complex_duplex['pathology'].append(0)
    print("Norma image " + name + " done.")
    print(str(len(norma_imgs_names) - norma_imgs_names.index(name)) + " norma images left.")

df_glcm_0 = pd.DataFrame(dict_glcm_0)
df_glcm_0.to_csv('glcm_0.csv', sep='\t', encoding='utf-8', index=False)

df_glcm_45 = pd.DataFrame(dict_glcm_45)
df_glcm_45.to_csv('glcm_45.csv', sep='\t', encoding='utf-8', index=False)

df_glcm_90 = pd.DataFrame(dict_glcm_90)
df_glcm_90.to_csv('glcm_90.csv', sep='\t', encoding='utf-8', index=False)

df_glcm_135 = pd.DataFrame(dict_glcm_135)
df_glcm_135.to_csv('glcm_135.csv', sep='\t', encoding='utf-8', index=False)

# df_glcm_complex = pd.DataFrame(dict_glcm_complex)
# df_glcm_complex.to_csv('glcm_complex.csv', sep='\t', encoding='utf-8', index=False)

# df_glcm_duplex = pd.DataFrame(dict_glcm_gen_duplex)
# df_glcm_duplex.to_csv('glcm_duplex.csv', sep='\t', encoding='utf-8', index=False)

# df_glcm_complex_duplex = pd.DataFrame(dict_glcm_complex_duplex)
# df_glcm_complex_duplex.to_csv('glcm_complex_duplex.csv', sep='\t', encoding='utf-8', index=False)



# print(df_glcm_0)
# g = glcm.glcm_complex()

# p = count_probabilities_arr(g)
# #

