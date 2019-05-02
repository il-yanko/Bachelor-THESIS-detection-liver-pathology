import numpy as np
import scipy.stats as stats

def average_RGB(img, windW, windH):
    #укрупним области
    w = len(img[0])
    h = len(img)
    ed = img
    for i in range(h-(windH-1)):
        for j in range(w-(windW-1)):
            R,G,B = 0,0,0
            for a in range (windH):
                for b in range (windW):
                    R += ed[i+a][j+b][0]
                    G += ed[i+a][j+b][1]
                    B += ed[i+a][j+b][2]
            size = windW * windH
            R /= size
            G /= size
            B /= size
            for a in range (windH):
                for b in range (windW):
                    ed[i+a][j+b][0] = R
                    ed[i+a][j+b][1] = G
                    ed[i+a][j+b][2] = B
    return ed
def average_gray(img, windW, windH):
    #укрупним области
    w = len(img[0])
    h = len(img)
    ed = img
    for i in range(h-(windH-1)):
        for j in range(w-(windW-1)):
            G = 0
            for a in range (windH):
                for b in range (windW):
                    G += ed[i+a][j+b]
            size = windW * windH
            G /= size
            for a in range (windH):
                for b in range (windW):
                    ed[i+a][j+b] = G
    return ed
# normalization to the sum
def normalize_array_2D(table):
    # division of each cell by the sum of all cells
    size1 = len(table)
    size2 = len(table[0])
    sum = 0
    rez = np.zeros([size1,size2],dtype=float)
    for i in range (size1):
        for j in range (size2):
            sum += table[i][j]
    for i in range(size1):
        for j in range(size2):
            rez[i][j] = table[i][j]/sum
    return rez
# compute descriptive statistics
def calculate_first_order_statistic_2D(data):
    agg_measures = {
        'avg': np.mean(data),
        'std': np.std(data),
        'var': np.var(data),
        'med': np.median(data),
        '10p': np.percentile(data, 10),
        '25p': np.percentile(data, 25),
        '50p': np.percentile(data, 50),
        '75p': np.percentile(data, 75),
        '90p': np.percentile(data, 90),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'skw': stats.skew(data.flatten()),
        'kur': stats.kurtosis(data.flatten())
    }
    return agg_measures
param = ['avg','std','var','med','10p','25p','50p','75p','90p','iqr','skw','kur']
paramNumber = len(param)


def check_dimension(testlist, dim=0):
   """tests if testlist is a list and how many dimensions it has
   returns -1 if it is no list at all, 0 if list is empty
   and otherwise the dimensions of it"""
   if isinstance(testlist, list):
      if testlist == []:
          return dim
      dim = dim + 1
      dim = check_dimension(testlist[0], dim)
      return dim
   else:
      if dim == 0:
          return -1
      else:
          return dim

# tests for check_dimension()
'''
a=[]
print (check_dimension(a))
a=""
print (check_dimension(a))
a=["A"]
print (check_dimension(a))
a=["A", "B", "C"]
print (check_dimension(a))
a=[[1,2,3],[1,2,3]]
print (check_dimension(a))
a=[[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]]
print (check_dimension(a))
'''