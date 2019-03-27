import numpy as np


class gradient:
    """This class implements the math of gradients"""

    # todo other types of gradients
    def __init__(self, array):
        self.base = array
        self.h, self.w = np.shape(array)

    def computeHorizontal(self):
        self.horizontal = np.zeros((self.h, self.w - 1))
        for i in range(self.h):
            for j in range(self.w - 1):
                self.horizontal[i][j] = abs(self.base[i][j] - self.base[i][j + 1])
        print(self.horizontal)

    def getHorizontal(self):
        return self.horizontal


"""
#test:
np.random.seed(1000)
A = np.around(np.random.random((5,3)),decimals=1)
B = gradient(A)
B.compute()
"""
