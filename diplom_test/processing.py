import numpy as np
import scipy.stats as stats


class NotAppropriateDimensionException(Exception):
    pass


def average_rgb(img, wind_w, wind_h):
    # it is blurring the RGB image
    if not check_dimension(img, 3):
        return
    w = len(img[0])
    h = len(img)
    ed = img
    for i in range(h-(wind_h - 1)):
        for j in range(w-(wind_w - 1)):
            r, g, b = 0, 0, 0
            for a in range(wind_h):
                for b in range(wind_w):
                    r += ed[i+a][j+b][0]
                    g += ed[i+a][j+b][1]
                    b += ed[i+a][j+b][2]
            size = wind_w * wind_h
            r /= size
            g /= size
            b /= size
            for a in range(wind_h):
                for b in range(wind_w):
                    ed[i+a][j+b][0] = r
                    ed[i+a][j+b][1] = g
                    ed[i+a][j+b][2] = b
    return ed


def average_gray(img, wind_w, wind_h):
    # it is blurring the gray image
    if not check_dimension(img, 2):
        return
    w = len(img[0])
    h = len(img)
    edit = img
    for i in range(h-(wind_h - 1)):
        for j in range(w-(wind_w - 1)):
            g = 0
            for a in range(wind_h):
                for b in range(wind_w):
                    g += edit[i+a][j+b]
            size = wind_w * wind_h
            g /= size
            for a in range(wind_h):
                for b in range(wind_w):
                    edit[i+a][j+b] = g
    return edit


# normalization to the sum
def normalize_2d_to_sum(table):
    if not check_dimension(table, 2):
        return
    # division of each cell by the sum of all cells
    size1 = len(table)
    size2 = len(table[0])
    cur_sum = 0
    rez = np.zeros([size1, size2], dtype=float)
    for i in range(size1):
        for j in range(size2):
            cur_sum += table[i][j]
    for i in range(size1):
        for j in range(size2):
            rez[i][j] = table[i][j]/cur_sum
    return rez


# compute descriptive statistics
def calculate_first_order_statistic_2d(data):
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


param = ['avg', 'std', 'var', 'med', '10p', '25p', '50p', '75p', '90p', 'iqr', 'skw', 'kur']
paramNumber = len(param)


def get_dimension(testlist, dim=0):
    """
    tests if testlist is a list and how many dimensions it has
    returns -1 if it is no list at all, 0 if list is empty
    and otherwise the dimensions of it
    """
    if isinstance(testlist, list):
        if testlist == []:
            return dim
        dim = dim + 1
        dim = get_dimension(testlist[0], dim)
        return dim
    else:
        if dim == 0:
            return -1
        else:
            return dim


def check_dimension(testlist, dim):

    try:
        cur_dim = get_dimension(testlist)
        if cur_dim != dim:
            raise NotAppropriateDimensionException
        return True
    except NotAppropriateDimensionException:
        print("Error! The received array is not",str(dim)+"-dimensional, but it is",str(cur_dim)+"-dimensional")
        return False
# tests for get_dimension()


'''
a=[]
print (get_dimension(a))
a=""
print (get_dimension(a))
a=["A"]
print (get_dimension(a))
a=["A", "B", "C"]
print (get_dimension(a))
a=[[1,2,3],[1,2,3]]
print (get_dimension(a))
a=[[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]]
print (get_dimension(a))
'''
# tests for check_dimension()
'''
a=[[1,2,3],[1,2,3]]
a=normalize_2D_to_sum(a)
print(a)
b=[[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]]
b=normalize_2D_to_sum(b)
print(b)
'''
# tests for average_gray()
'''
a=[[1,100,1],[100,1,100]]
a=average_gray(a,2,2)
print(a)
b=[[[1,100,1],[100,1,100]]]
b=average_gray(b,2,2)
print(b)
'''