import os.path
import time
from builtins import print
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import cm
from builtins import print, complex
import matplotlib
import pylab
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import pygame, sys
from PIL import Image
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from tkinter import Tk, Button
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image, ImageTk

print(time.localtime())


def change_matrix(matrix):
    changed_matrix = np.zeros((len(matrix) - 1, len(matrix[1]) - 1), dtype=int)
    for i in range(len(matrix) - 1):
        for j in range(len(matrix[i]) - 1):
            changed_matrix[i][j] = int(matrix[i][j]) + int(matrix[i + 1][j]) + \
                                   int(matrix[i][j + 1]) + int(
                matrix[i + 1][j + 1])
    return changed_matrix


def fff(matrix):
    # new_matrix = change_matrix(matrix)
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


def check_data(dir, data_file):
    with open(data_file) as file:
        value_scale = [row.strip() for row in file]
    filename = [name for name in os.listdir(dir) if
                os.path.isfile(os.path.join(dir, name))]
    # count_file = len(filename)
    return value_scale, filename


def sort_first(val):
    return val[0]


def choice_limit(value_scale):
    scale_array = []
    limit_array = []
    index_array = []
    for value in value_scale:
        scale_array.append([int(value.split(" ")[0]), int(value.split(" ")[1])])
    scale_array.sort(key=sort_first)

    scale_array = np.array(scale_array)
    for limit in range(len(scale_array)):
        limit_array.append(len(np.where((scale_array[:, 1] == 1) & (scale_array[:, 0] <= scale_array[limit][0]))[0]) +
                           len(np.where((scale_array[:, 1] == 2) & (scale_array[:, 0] >= scale_array[limit][0]))[0]))
    max_lim = max(limit_array)
    limit_array2 = np.array(limit_array)
    limit_array2 = np.where(limit_array2 == max(limit_array2))[0]
    index_lim = len(limit_array2) // 2
    [index_array.append(idx) for idx, num in enumerate(limit_array) if num == max(limit_array)]
    limit = (scale_array[index_array[index_lim]][0] + scale_array[index_array[index_lim] + 1][0]) // 2
    f = open("limit_1.txt", 'w')
    f.write("{}".format(limit))
    f.close()
    # return limit
    # print(int(len(np.where(limit_array == max(limit_array))[0])//2))


def update_data(direct, data_file):
    result_check = check_data(direct, data_file)
    value_scale = result_check[0]
    filename = result_check[1]
    count_file = len(filename)
    if len(value_scale) != count_file:
        f = open(data_file, 'w')
        for name in filename:
            if name.find('norm') >= 0:
                image_class = 1
            else:
                image_class = 2

            # Read the main image
            img_rgb = cv2.imread('{}'.format(name))
            # Convert it to grayscale
            matrix = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            # matrix = matrix[:15, :10]

            res = fff(matrix)
            tp1 = res[0]
            tp2 = res[1]
            value = res[2]
            max_index = value.index(max(value))

            result = int(tp1[max_index]) * int(tp2[max_index])
            f.write('{} {}\n'.format(result, image_class))

            # dz = value
            # cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
            # max_height = np.max(dz)  # get range of colorbars so we can normalize
            # min_height = np.min(dz)
            # # scale each z to [0,1], and get their rgb values
            # rgba = [cmap((k - min_height) / max_height) for k in dz]
            # # print(tp1)
            # # print(tp2)
            # # print(dz)
            # # print(collections.OrderedDict(sorted(Counter(dz).items())))
            # # print(rgba)
            # fig = plt.figure()
            #
            # ax = fig.add_subplot(111, projection='3d')
            # ax.bar3d(0, 0, 0, tp1, tp2, value, color=rgba)
            # # plt.title("Патология{}".format(i))
            # plt.show()
            #
            # # res = fff(matrix)
            # # # img = fff( res)
            # # for i in range(len(tp1)):
            # #     if tp1[i] < 37 and tp1[i] >20 and tp2[i] < 37 or
            # #         tp1[i] < 37 and tp2[i] > 20 and tp2[i] < 37
            # res = res[3]
            # # res = preprocessing.normalize(res, norm='l2') * 255
            # # res = np.int64(res)
            # #
            # # img = preprocessing.normalize(img, norm='l2') * 255
            # # img = np.int64(img)
            #
            print(time.localtime())
            #
            # #
            # # plt.title("Исходные данные")
            # # plt.imshow(matrix, cmap='gray')
            # # plt.show()
            # # plt.title("Препроцессинг")
            # # plt.imshow(change_matrix(matrix), cmap='gray')
            # # plt.show()
            # # plt.title("Первый проход")
            # plt.imshow(res)  # , cmap='gray')
            # plt.show()
            # # plt.title("Второй проход проход")
            # # plt.imshow(img, cmap='gray')
            # # plt.show()
        f.close()
        update_data(direct, data_file_1)
    if len(check_data(direct, "limit_1.txt")[0]) == 0:
        choice_limit(value_scale)


def final():
    global direct, data_file_1, matrix
    update_data(direct, data_file_1)
    res = fff(matrix)
    tp1 = res[0]
    tp2 = res[1]
    value = res[2]
    plot_hist(tp1, tp2, value)
    max_index = value.index(max(value))
    result = int(tp1[max_index]) * int(tp2[max_index])
    color = 'g'
    scale_array = []
    with open(data_file_1) as file:
        value_scale = [row.strip() for row in file]
    for value in value_scale:
        scale_array.append([int(value.split(" ")[0]), int(value.split(" ")[1])])
    max_sc = max(np.array(scale_array)[:, 0])
    if int(result) > int(check_data(direct, "limit_1.txt")[0][0]):
        color = 'r'
    x = [1, 2]
    y = [int(check_data(direct, "limit_1.txt")[0][0]) * 100 / max_sc, result * 100 / max_sc]
    fig = plt.figure()
    barlist = plt.bar(x, y)
    barlist[0].set_color('g')
    barlist[1].set_color(color)
    plt.xticks(x, ("Норма", "Загруженное изображение"))
    plt.show()


def plot_hist(tp1, tp2, value):
    cmap = cm.get_cmap('jet')
    max_height = np.max(value)
    min_height = np.min(value)
    rgba = [cmap((k - min_height) / max_height) for k in value]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(0, 0, 0, tp1, tp2, value, color=rgba)
    # plt.show()


# f.close()





def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width = pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)


def setup(path):
    px = pygame.image.load(path)
    screen = pygame.display.set_mode(px.get_rect()[2:])
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px


def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n = 0
    while n != 1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n = 1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return (topleft + bottomright)


if __name__ == "__main__":
    pygame.init()
    Tk().withdraw()
    filename = askopenfilename(filetypes=(("bmp file", "*.bmp"), ("jpg file", "*.jpg"), ("All Files", "*.*")))
    direct = 'Image'
    data_file_1 = "data_1.txt"
    # delta = 0
    # Read the main image

    input_loc = filename
    output_loc = 'out.bmp'
    screen, px = setup(input_loc)
    left, upper, right, lower = mainLoop(screen, px)

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower
    im = Image.open(input_loc)
    im = im.crop((left, upper, right, lower))
    pygame.display.quit()
    im.save(output_loc)
    img_rgb = cv2.imread('out.bmp')
    # Convert it to grayscale
    matrix = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    final()
    # pygame.display.quit()
