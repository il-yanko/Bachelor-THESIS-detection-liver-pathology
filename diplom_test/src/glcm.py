import numpy as np

class GLCM:
    """GLCM = Gray-Level Co-Occurrence Matrix"""

    def __init__(self, gray):
        self.grey = gray
        self.h, self.w = np.shape(gray)
        self.size = 255

    def glcm_0(self):
        """horizontal co-occurrence"""
        glcm = np.zeros([self.size, self.size], dtype=int)
        enabled = np.zeros([self.size, self.size])
        for i in range(self.h):
            for j in range(self.w - 1):
                left = self.grey[i][j]
                right = self.grey[i][j + 1]
                if enabled[left - 1][right - 1] != 1:
                    for a in range(self.h):
                        for b in range(self.w - 1):
                            newLeft = self.grey[a][b]
                            newRight = self.grey[a][b + 1]
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
                leftLow = self.grey[i + 1][j]
                rightUp = self.grey[i][j + 1]
                if enabled[leftLow - 1][rightUp - 1] != 1:
                    for a in range(self.h - 1):
                        for b in range(self.w - 1):
                            newLeftLow = self.grey[a + 1][b]
                            newRightUp = self.grey[a][b + 1]
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
                up = self.grey[i][j]
                low = self.grey[i + 1][j]
                if enabled[up - 1][low - 1] != 1:
                    for a in range(self.h - 1):
                        for b in range(self.w):
                            newLow = self.grey[a + 1][b]
                            newUp = self.grey[a][b]
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
                leftUp = self.grey[i][j]
                rightLow = self.grey[i + 1][j + 1]
                if enabled[leftUp - 1][rightLow - 1] != 1:
                    for a in range(self.h - 1):
                        for b in range(self.w - 1):
                            newLeftUp = self.grey[a][b]
                            newRightLow = self.grey[a + 1][b + 1]
                            if newLeftUp == leftUp and newRightLow == rightLow:
                                glcm[leftUp - 1][rightLow - 1] += 1
                    enabled[leftUp - 1][rightLow - 1] = 1
        return glcm

    def glcm_complex(self):
        """the sum of all kinds of co-occurrence"""
        return self.glcm_0() + self.glcm_45() + self.glcm_90() + self.glcm_135()

    def glcm_gen_duplex(self, method):
        """to equalize pairs A-B and B-A and to get the result"""
        glcm = method()  # it should obligatory be a glcm-like kind of the method
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