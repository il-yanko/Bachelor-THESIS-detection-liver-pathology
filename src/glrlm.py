import numpy as np


class GLRLM:
    """
    GLRLM - Gray level run length matrix
    """

    def __init__(self, image):
        self.image = image
        self.h, self.w = np.shape(image)
        self.gray_level = 4
        self.glrlm = None


    def glrlm_0(self):
        """Vertical run"""

        glrlm = np.zeros([self.gray_level, self.w], dtype=int)

        for i in range(self.h):
            count = 0
            for j in range(self.w):
                if j < self.w - 1 and self.image[i][j] == self.image[i][j + 1]:
                    count += 1
                else:
                    glrlm[self.image[i][j]][count] += 1
                    count = 0
        return glrlm

    def LGRE(self, glrlm):
        res = 0
        s = glrlm.shape[1]

        for i in range(glrlm.shape[0]):
            for j in range(glrlm.shape[1]):
                res += (glrlm[i][j] / s) / (i * i) if i != 0 else 0

        return res

    def HGRE(self, glrlm):
        res = 0
        s = glrlm.shape[1]

        for i in range(glrlm.shape[0]):
            for j in range(glrlm.shape[1]):
                res += (glrlm[i][j] * (i * i)) / s

        return res

    def GLNU(self, glrlm):
        # res = 0
        s = glrlm.shape[1]

        res = sum([sum([g**2 for g in line]) for line in glrlm]) / s
        # for i in range(glrlm.shape[0]):
        #     for j in range(glrlm.shape[1]):
                # res += glrlm[i][j]**2

        return res


image = np.asarray([
    [0, 1, 2, 3],
    [0, 2, 3, 3],
    [2, 1, 1, 1],
    [3, 0, 3, 0]
])

gl = GLRLM(image)

print(gl.glrlm_0())
res = gl.glrlm_0()

print("LGRE: ", gl.LGRE(res))
print("HGRE: ", gl.HGRE(res))
print("GLNU: ", gl.GLNU(res))
