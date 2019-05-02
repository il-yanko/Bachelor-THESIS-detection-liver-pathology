import numpy as np
import matplotlib
from processing import check_dimension

class GLCM:
    """
    GLCM = Gray-Level Co-Occurrence Matrix
    """
    def __init__(self, gray):
        if (not check_dimension(gray, 2)):
            raise ValueError("It was received not appropriate dimension!")
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
                            newLeft = self.image[a][b]
                            newRight = self.image[a][b + 1]
                            if newLeft == left and newRight == right:
                                glcm[left - 1][right - 1] += 1
                    enabled[left - 1][right - 1] = 1
        return glcm

    def glcm_45(self):
        """diagonal from the left down to the right upper corner co-occurrence"""
        glcm = np.zeros([self.size, self.size], dtype=int)
        enabled = np.zeros([self.size, self.size])
        for i in range(self.h - 1):
            for j in range(self.w - 1):
                leftLow = self.image[i + 1][j]
                rightUp = self.image[i][j + 1]
                if enabled[leftLow - 1][rightUp - 1] != 1:
                    for a in range(self.h - 1):
                        for b in range(self.w - 1):
                            newLeftLow = self.image[a + 1][b]
                            newRightUp = self.image[a][b + 1]
                            if newLeftLow == leftLow and newRightUp == rightUp:
                                glcm[leftLow - 1][rightUp - 1] += 1
                    enabled[leftLow - 1][rightUp - 1] = 1
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
                            newLow = self.image[a + 1][b]
                            newUp = self.image[a][b]
                            if (newLow == low and newUp == up):
                                glcm[up - 1][low - 1] += 1
                    enabled[up - 1][low - 1] = 1
        return glcm

    def glcm_135(self):
        """diagonal from the left upper to the right down corner co-occurrence"""
        glcm = np.zeros([self.size, self.size], dtype=int)
        enabled = np.zeros([self.size, self.size])
        for i in range(self.h - 1):
            for j in range(self.w - 1):
                leftUp = self.image[i][j]
                rightLow = self.image[i + 1][j + 1]
                if enabled[leftUp - 1][rightLow - 1] != 1:
                    for a in range(self.h - 1):
                        for b in range(self.w - 1):
                            newLeftUp = self.image[a][b]
                            newRightLow = self.image[a + 1][b + 1]
                            if newLeftUp == leftUp and newRightLow == rightLow:
                                glcm[leftUp - 1][rightLow - 1] += 1
                    enabled[leftUp - 1][rightLow - 1] = 1
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