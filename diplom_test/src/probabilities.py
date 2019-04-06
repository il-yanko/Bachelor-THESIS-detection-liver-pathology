from sklearn.preprocessing import normalize
from glcm import GLCM
from img_reader import IMGReader
import numpy as np

PROJECT_PWD = "/Volumes/Storage/goinfre/ikachko/algorithm-detects-liver-pathology/diplom_test"

NORMA_DIR = PROJECT_PWD + "/norma/"
PATHOLOGY_DIR = PROJECT_PWD + "/pathology/"

img_names, images = IMGReader.read_directory(NORMA_DIR)

# def normalize_color(matrix):
#     for row
#     return normalize(matrix, axis)

# print(type(images[0]))

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

    def energy(self):
        return np.sum(np.power(self.matrix.flatten(), 2))

glcm = GLCM(images[0])

g = glcm.glcm_complex()

p = count_probabilities_arr(g)

gleq = GLCMEquations(p)
print("Energy: ", gleq.energy())
