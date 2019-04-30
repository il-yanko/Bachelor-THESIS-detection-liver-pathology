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

# garbage processing
# PE works but have NO sense
'''
# PERMUTATION ENTROPY and ordinal patterns calculations and processing
patternNumber = 5
#min, max, increasing, decreasing, stability
patternAXIS1 = 1
patternAXIS2 = 4
AXIS1name = "МАКСИМУМ, %"
AXIS2name = "СТАБІЛЬНІСТЬ, %"

print('\nPathological specimens')
pathoPEpatterns = np.zeros((1, patternNumber))
for i in range(len(pathoBMP)):
    tmp = pe.pe(pathoBMP[i])
    tmp.PErawAnalysis(10)
    # это чтобы добавить 1 (!!!) отметку в легенду, мб исправить потом
    # НАДО перейти к scatter (!!!)
    plt.plot(tmp.rez[0][patternAXIS1],
             tmp.rez[0][patternAXIS2],
             'ro')
    #label='pathological SAMPLES'
    for j in range(patternNumber):
        pathoPEpatterns[0][j] += tmp.rez[0][j]

for k in range(patternNumber):
    pathoPEpatterns[0][k] /= len(pathoBMP)
plt.plot(pathoPEpatterns[0][patternAXIS1],
         pathoPEpatterns[0][patternAXIS2],
         'r*', label='pathological AVERAGE',
         markersize=30)
print(pathoPEpatterns)

print('Normal specimens')
normaPEpatterns = np.zeros((1, patternNumber))
for i in range(len(normaBMP)):
    tmp = pe.pe(normaBMP[i])
    tmp.PErawAnalysis(10)
    plt.plot(tmp.rez[0][patternAXIS1],
             tmp.rez[0][patternAXIS2],
             'b^')
    for j in range(patternNumber):
        normaPEpatterns[0][j] += tmp.rez[0][j]

for k in range(patternNumber):
    normaPEpatterns[0][k] /= len(normaBMP)
plt.plot(normaPEpatterns[0][patternAXIS1],
         normaPEpatterns[0][patternAXIS2],
         'b*', label='normal AVERAGE',
         markersize=30)
print(normaPEpatterns)

perc = np.zeros((1, patternNumber))
for i in range(patternNumber):
    dif = abs(normaPEpatterns[0][i] - pathoPEpatterns[0][i])
    perc[0][i] = (dif / max(normaPEpatterns[0][i], pathoPEpatterns[0][i])) * 100
print('Normalized difference in percents:')
print(perc)

plt.xlabel(AXIS1name)
plt.ylabel(AXIS2name)
plt.legend()
plt.savefig('ordinal_patterns.png')
'''
