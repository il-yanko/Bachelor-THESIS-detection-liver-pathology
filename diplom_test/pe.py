import numpy as np
from pyentrp import entropy as ent


# permutation entropy
def greyPicPE(picture, normalize, order=3, delay=1):
    length = len(picture)
    PEnt = np.zeros(length)
    for i in range(length):
        tempStr = picture[i]
        PEnt[i] = ent.permutation_entropy(tempStr, order, delay, normalize)
    return (PEnt)


def calculateDiffPEvalues(BMP):
    lengh = len(BMP)
    PEnormalized = np.zeros(lengh)
    PEmid = np.zeros(lengh)
    for i in range(lengh):
        currentPicPE = greyPicPE(BMP[i], True, 5, 1)
        PEnormalized[i] = np.round(ent.permutation_entropy(currentPicPE, 4, 1, True), 3)
        PEmid[i] = np.mean(currentPicPE).round(3)
    print('The normalized permutation entropy:\n', PEnormalized)
    print('The mean permutation entropy:\n', PEmid, '\n')
    return PEnormalized, PEmid
