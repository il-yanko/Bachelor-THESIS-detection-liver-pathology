import numpy as np
from pyentrp import entropy as ent
import frp


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

    '''
    # not valid = для многих картинок в старой версии
    def calculateDiffPEvalues(self, order=3, delay=1):
        length = len(BMP)
        PEnormalized = np.zeros(length)
        PEmid = np.zeros(length)
        for i in range(length):
            currentPicPE = basicalPE1D(BMP[i], True, order, delay)
            PEnormalized[i] = np.round(ent.permutation_entropy(currentPicPE, 4, 1, True), 3)
            PEmid[i] = np.mean(currentPicPE).round(3)
        print('The normalized permutation entropy:\n', PEnormalized)
        print('The mean permutation entropy:\n', PEmid, '\n')
        return PEnormalized, PEmid
    '''

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
            # print(curPicPE)
            particle = frp.tripleEvaluator(curPicPE[j], curPicPE[j + 1], curPicPE[j + 2])
            particle.checkConditions()
            tmp = particle.getResults()
            # print(tmp)
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


'''
a = [[13, 3, 10, 12, 7, 13, 8, 11, 17, 9],
     [10, 4, 20, 9, 14, 1, 20, 14, 18, 8],
     [6, 17, 18, 15, 12, 16, 5, 14, 3, 18]]
b = pe(a)
# changed, now it does not return values
# print( b.PErawAnalysis() )
'''
