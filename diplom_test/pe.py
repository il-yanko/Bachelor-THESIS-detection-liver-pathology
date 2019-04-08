import numpy as np
from pyentrp import entropy as ent
import fop


class pe:
    def __init__(self, array):
        self.base = array
        self.length = len(self.base)
        self.rez = np.zeros((1, 5))

    def basicalPE1D(self, normalize, order=3, delay=1):
        PEnt = np.zeros(self.length)
        for i in range(self.length):
            tempStr = self.base[i]
            PEnt[i] = ent.permutation_entropy(tempStr, order, delay, normalize)
        return (PEnt)

    def PErawAnalysis(self, order=3, delay=1):
        """
        we will calculate the number of forms of classes
        of our regulatory patterns such as:
        min, max, increasing, decreasing, stability
        """
        # current picture's PE calculations
        curPicPE = self.basicalPE1D(True, order, delay)
        minC = 0
        maxC = 0
        ascC = 0
        desC = 0
        stbC = 0
        for j in range(len(curPicPE) - 2):
            particle = fop.tripleEvaluator(curPicPE[j], curPicPE[j + 1], curPicPE[j + 2])
            particle.checkConditions()
            tmp = particle.getResults()
            if (tmp[0]):
                minC += 1
                continue
            if (tmp[1]):
                maxC += 1
                continue
            if (tmp[2]):
                ascC += 1
                continue
            if (tmp[3]):
                desC += 1
                continue
            if (tmp[4]):
                stbC += 1
        types = minC + maxC + ascC + desC + stbC
        self.rez[0][0] = minC / types
        self.rez[0][1] = maxC / types
        self.rez[0][2] = ascC / types
        self.rez[0][3] = desC / types
        self.rez[0][4] = stbC / types