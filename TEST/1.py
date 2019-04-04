import os.path
import cv2
from collections import Counter
import numpy as np
import pandas as pd


def check_file(dir):
    filename = [name for name in os.listdir(dir) if
                os.path.isfile(os.path.join(dir, name))]
    return filename


def fff(matrix):
    tp1 = []
    tp2 = []
    value = []
    new_matrix = matrix
    new_array = np.array(new_matrix.ravel())
    final_matrix = np.zeros((len(new_matrix), len(new_matrix[1])), dtype=int)
    tp = []
    counter = 0
    for idx in range(len(new_array) - 1):

        count = 0
        pair = np.array((new_array[idx], new_array[idx + 1]))
        for i in range(len(tp)):
            if abs(int(tp[i][0][0]) - int(pair[0])) <= 0 and abs(
                    int(tp[i][0][1]) - int(pair[1])) <= 0:
                count = tp[i][1]
                break

        if count == 0:
            count = count + sum(1 for k in [new_array[j:j + len(pair)] for j in
                                            range(len(new_array) - len(
                                                pair) + 1)]
                                if abs(int(k[0]) - int(pair[0])) <= 0 and
                                abs(int(k[1]) - int(pair[1])) <= 0)
            tp.append([pair, count])
            tp1.append(tp[counter][0][0])
            tp2.append(tp[counter][0][1])
            value.append(tp[counter][1])
            counter += 1
        if (tp1[counter - 1] < 50 and tp1[counter - 1] > 15 and tp2[
            counter - 1] <
            0) or (tp1[counter - 1] < 50 and tp2[counter - 1] > 15 and tp2[
            counter - 1] < 50):
            # final_matrix[idx // len(new_matrix[0])][idx % len(new_matrix[
            #                                                       0])] = 255
            # value[counter-1] = 50
            final_matrix[idx // len(new_matrix[0])][idx % len(new_matrix[
                                                                  0])] = \
                new_array[idx]
        else:
            final_matrix[idx // len(new_matrix[0])][idx % len(new_matrix[
                                                                  0])] = 255

    return tp1, tp2, value, final_matrix[:-1][:]


def optimal_value_second():
    result_check = check_file("Image")
    # for i in range(50, 71):
    array1 = []
    for value_gray in range(50,70):

        array2 = []
        for name in result_check:
            # name = "norm1.bmp"
            if name.find('norm') >= 0:
                image_class = 1
            else:
                image_class = 2
            print(name)
            # Read the main image
            img_rgb = cv2.imread('{}'.format(name))
            # Convert it to grayscale
            matrix = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            res = fff(matrix)
            tp1 = res[0]
            tp2 = res[1]
            value = pd.Series(res[2])
            value = (value / sum(value) * 100000).tolist()
            # value = res[2] / (sum(res[2]) * 100)
            try:
                res2 = value[tp1.index(value_gray)]

                array2.append([res2, image_class])
            except:
                pass
        array1.append([choice_limit(array2)[1],value_gray])
        print(array1)
    return array1


def sort_first(val):
    return val[0]


def choice_limit(value_scale):
    scale_array = []
    limit_array = []
    index_array = []
    # for value in value_scale:
    #     scale_array.append([int(value.split(" ")[0]), int(value.split(" ")[1])])


    # scale_array = np.array(scale_array)
    scale_array = np.array(value_scale)
    # scale_array.sort(key=sort_first)

    for limit in range(len(scale_array)):
        limit_array.append(len(np.where((scale_array[:, 1] == 1) & (scale_array[:, 0] <= scale_array[limit][0]))[0]) +
                           len(np.where((scale_array[:, 1] == 2) & (scale_array[:, 0] >= scale_array[limit][0]))[0]))
    limit_array2 = np.array(limit_array)
    return_value = max(limit_array2)
    limit_array2 = np.where(limit_array2 == max(limit_array2))[0]
    index_lim = len(limit_array2) // 2
    [index_array.append(idx) for idx, num in enumerate(limit_array) if num == max(limit_array)]
    limit = (scale_array[index_array[index_lim]][0] + scale_array[index_array[index_lim] + 1][0]) // 2
    return limit, return_value

res = optimal_value_second()
print(res)

# print(choice_limit([[7.271216370624285, 1], [0.34204405527431935, 1], [1.0291242152927859, 1], [1.6304237063606904, 1], [2.9568302779420463, 1], [8.231420507996237, 1], [1.0880603510808065, 1], [3.448672261179446, 1], [1.0423181154888472, 1], [6.839165621794141, 2], [2.6167735182519955, 2], [21.06002106002106, 2], [13.71742112482853, 2], [7.557791040108415, 2], [2.5482035165208528, 2], [0.4131264029084099, 2], [1.3605627287446088, 2], [5.9844404548174746, 2], [7.84313725490196, 2]]))

# a =[[7.271216370624285, 1], [0.34204405527431935, 1], [1.0291242152927859, 1], [1.6304237063606904, 1], [2.9568302779420463, 1], [8.231420507996237, 1], [1.0880603510808065, 1], [3.448672261179446, 1], [1.0423181154888472, 1], [6.839165621794141, 2], [2.6167735182519955, 2], [21.06002106002106, 2], [13.71742112482853, 2], [7.557791040108415, 2], [2.5482035165208528, 2], [0.4131264029084099, 2], [1.3605627287446088, 2], [5.9844404548174746, 2], [7.84313725490196, 2]]