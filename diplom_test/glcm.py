import numpy as np
import matplotlib.pyplot as plt


class GLCM:
    """
    GLCM = Gray-Level Co-Occurrence Matrix
    """
    def __init__(self, gray):
        self.image = gray
        self.h, self.w = np.shape(gray)
        self.size = 255
        self.saved = gray

    def glcm_0(self):
        """horizontal co-occurrence"""
        glcm = np.zeros([self.size, self.size], dtype=int)
        enabled = np.zeros([self.size, self.size])
        for i in range(self.h):
            for j in range(self.w - 1):
                left = self.image[i][j]
                right = self.image[i][j + 1]
                if enabled[left - 1][right - 1] != 1:
                    for a in range(self.h):
                        for b in range(self.w - 1):
                            new_left = self.image[a][b]
                            new_right = self.image[a][b + 1]
                            if new_left == left and new_right == right:
                                glcm[left - 1][right - 1] += 1
                    enabled[left - 1][right - 1] = 1
        return glcm

    def glcm_45(self):
        """diagonal from the left down to the right upper corner co-occurrence"""
        glcm = np.zeros([self.size, self.size], dtype=int)
        enabled = np.zeros([self.size, self.size])
        for i in range(self.h - 1):
            for j in range(self.w - 1):
                left_low = self.image[i + 1][j]
                right_up = self.image[i][j + 1]
                if enabled[left_low - 1][right_up - 1] != 1:
                    for a in range(self.h - 1):
                        for b in range(self.w - 1):
                            new_left_low = self.image[a + 1][b]
                            new_right_up = self.image[a][b + 1]
                            if new_left_low == left_low and new_right_up == right_up:
                                glcm[left_low - 1][right_up - 1] += 1
                    enabled[left_low - 1][right_up - 1] = 1
        return glcm

    def glcm_90(self):
        """vertical co-occurrence"""
        glcm = np.zeros([self.size, self.size], dtype=int)
        enabled = np.zeros([self.size, self.size])
        for i in range(self.h - 1):
            for j in range(self.w):
                up = self.image[i][j]
                low = self.image[i + 1][j]
                if enabled[up - 1][low - 1] != 1:
                    for a in range(self.h - 1):
                        for b in range(self.w):
                            new_low = self.image[a + 1][b]
                            new_up = self.image[a][b]
                            if new_low == low and new_up == up:
                                glcm[up - 1][low - 1] += 1
                    enabled[up - 1][low - 1] = 1
        return glcm

    def glcm_135(self):
        """diagonal from the left upper to the right down corner co-occurrence"""
        glcm = np.zeros([self.size, self.size], dtype=int)
        enabled = np.zeros([self.size, self.size])
        for i in range(self.h - 1):
            for j in range(self.w - 1):
                left_up = self.image[i][j]
                right_low = self.image[i + 1][j + 1]
                if enabled[left_up - 1][right_low - 1] != 1:
                    for a in range(self.h - 1):
                        for b in range(self.w - 1):
                            new_left_up = self.image[a][b]
                            new_right_low = self.image[a + 1][b + 1]
                            if new_left_up == left_up and new_right_low == right_low:
                                glcm[left_up - 1][right_low - 1] += 1
                    enabled[left_up - 1][right_low - 1] = 1
        return glcm

    def glcm_complex(self):
        """the sum of all kinds of co-occurrence"""
        return self.glcm_0() + self.glcm_45() + self.glcm_90() + self.glcm_135()

    def glcm_gen_duplex(self, method):
        """to equalize pairs A-B and B-A and to get the result"""
        # this function is required for creation of Color-Color symmetrical GLCM
        glcm = method()  # it should obligatory be a GLCM-like kind of the method
        for i in range(len(glcm)):
            for j in range(i + 1):
                if i == j:
                    glcm[i][j] *= 2
                    continue
                else:
                    result = glcm[i][j] + glcm[j][i]
                    glcm[i][j] = result
                    glcm[j][i] = result
        return glcm

    def glcm_complex_duplex(self):
        return self.glcm_gen_duplex(self.glcm_0) \
               + self.glcm_gen_duplex(self.glcm_45) \
               + self.glcm_gen_duplex(self.glcm_90) \
               + self.glcm_gen_duplex(self.glcm_135)


def calculate_save_glcm(short_name, array):
    for i in range(len(array)):
        cur_im = array[i]
        calculation = GLCM(cur_im).glcm_complex_duplex()
        number = i + 1
        print(number)
        # image
        path = "glcm/" + short_name + "/png/" + short_name + str(number) + ".png"
        plt.imsave(path, calculation, cmap="inferno")
        # data
        path = "glcm/" + short_name + "/csv/" + short_name + str(number) + ".csv"
        np.savetxt(path, calculation, fmt="%d", delimiter=",")
    print(short_name, " was saved successfully")


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

# garbage processing
# processing and saving of GLCM (.csv)
'''
for i in range(len(normaBMP)):
    curIm = normaBMP[i]
    calculation = glcm.GLCM(curIm).glcm_complex_duplex()
    number = i + 1
    print(number)
    path = "glcm/n/n" + str(number) + ".csv"
    np.savetxt(path, calculation, fmt="%d", delimiter=",")
for i in range(len(pathoBMP)):
    curIm = pathoBMP[i]
    calculation = glcm.GLCM(curIm).glcm_complex_duplex()
    number = i + 1
    print(number)
    path = "glcm/p/p" + str(number) + ".csv"
    np.savetxt(path, calculation, fmt="%d", delimiter=",")
for i in range(len(auhBMP)):
    curIm = auhBMP[i]
    calculation = glcm.GLCM(curIm).glcm_complex_duplex()
    number = i + 1
    print(number)
    path = "glcm/auh/auh" + str(number) + ".csv"
    np.savetxt(path, calculation, fmt="%d", delimiter=",")
'''
# 5x5 images
'''
columns = 5
rows    = math.ceil(len(normaBMP) / columns)
for i in range(len(normaBMP)):
    curIm = normaBMP[i]
    #curIm = curIm[0:3,0:3]
    #plt.imshow(curIm)
    curRow, curCol = 0, 0
    calculation = glcm.GLCM(curIm).glcm_complex_duplex()
    number = i+1
    print(rows,columns,number)
    fig.add_subplot(rows,columns,number)
    plt.imshow(calculation)
    plt.tight_layout()
plt.savefig('tmp/norma.png')
'''
# save all 50 GLCMs as temporary
'''
for i in range(len(normaBMP)):
    curIm = normaBMP[i]
    calculation = glcm.GLCM(curIm).glcm_complex_duplex()
    number = i+1
    print(number)
    plt.imshow(calculation)
    path = "glcm/n/n" + str(number) + ".png"
    plt.imsave(path, calculation, cmap="inferno")
print("norma was saved sucessfull")
for i in range(len(pathoBMP)):
    curIm = pathoBMP[i]
    calculation = glcm.GLCM(curIm).glcm_complex_duplex()
    number = i+1
    print(number)
    plt.imshow(calculation)
    path = "glcm/p/p" + str(number) + ".png"
    plt.imsave(path, calculation, cmap="inferno")
print("pathology was saved sucessfull")
'''