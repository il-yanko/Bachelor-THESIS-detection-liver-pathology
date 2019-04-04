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
# from matplotlib import cm
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


def change_matrix(matrix):
    changed_matrix = np.zeros((len(matrix) - 1, len(matrix[1]) - 1), dtype=int)
    for i in range(len(matrix) - 1):
        for j in range(len(matrix[i]) - 1):
            changed_matrix[i][j] = int(matrix[i][j]) + int(matrix[i + 1][j]) + \
                                   int(matrix[i][j + 1]) + int(
                matrix[i + 1][j + 1])
    return changed_matrix


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


def read_file(data_file):
    with open(data_file) as file:
        value_scale = [row.strip() for row in file]
    return value_scale


def check_file(dir):
    filename = [name for name in os.listdir(dir) if
                os.path.isfile(os.path.join(dir, name))]
    # count_file = len(filename)
    return filename


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
    limit_array2 = np.array(limit_array)
    limit_array2 = np.where(limit_array2 == max(limit_array2))[0]
    index_lim = len(limit_array2) // 2
    [index_array.append(idx) for idx, num in enumerate(limit_array) if num == max(limit_array)]
    limit = (scale_array[index_array[index_lim]][0] + scale_array[index_array[index_lim] + 1][0]) // 2
    return limit, index_lim


def update_data(direct, data_file):
    value_scale = read_file(data_file)
    filename = check_file(direct)
    count_file = len(filename)
    if len(value_scale) != count_file:
        f = open(data_file, 'w')
        f2 = open("data_2.txt", 'w')
        f3 = open("data_3.txt", 'w')
        f4 = open("data_4.txt", 'w')
        f5 = open("data_5.txt", 'w')
        for name in filename:
            if name.find('norm') >= 0:
                image_class = 1
            else:
                image_class = 2
            print(name)
            # Read the main image
            img_rgb = cv2.imread('{}'.format(name))
            # Convert it to grayscale
            matrix = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            # matrix = matrix[:15, :10]

            res = fff(matrix)
            # lenx = len(matrix)
            # leny = len(matrix[1])
            tp1 = res[0]
            tp2 = res[1]
            value = res[2]
            max_index = value.index(max(value))

            new_tp1 = max(sorted(tp1)[int(len(tp1) / 2):])
            result = int(tp1[max_index]) * int(tp2[max_index])

            medium_index = value.index(list(set(sorted(value)))[int(len(set(value)) / 2)])
            result2 = int(tp1[medium_index]) * int(tp2[medium_index])

            result3 = int(max(tp1)) * int(max(tp2))  # необходимо  добавить в финал

            diagonal = []
            medium_elem = list(set(sorted(tp1)))[int(len(set(tp1)) / 2)]
            for el in range(len(tp1)):
                if tp1[el] >= medium_elem:
                    diagonal.append(value[el])
            result4 = max(diagonal)

            medium_scale_index = tp1.index(list(set(sorted(tp1)))[int(len(set(tp1)) / 2)])
            result5 = value[medium_scale_index]

            f.write('{} {}\n'.format(result, image_class))
            f2.write('{} {}\n'.format(result2, image_class))
            f3.write('{} {}\n'.format(result3, image_class))
            f4.write('{} {}\n'.format(result4, image_class))
            f5.write('{} {}\n'.format(result5, image_class))

        f.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()

        update_data(direct, data_file)
    if len(read_file("limit_1.txt")) == 0:
        f = open("limit_1.txt", 'w')
        f.write("{}".format(choice_limit(value_scale)))
        f.close()
    if len(read_file("limit_2.txt")) == 0:
        f = open("limit_2.txt", 'w')
        f.write("{}".format(choice_limit(read_file("limit_2.txt")[0])))
        f.close()
    if len(read_file("limit_3.txt")) == 0:
        f = open("limit_3.txt", 'w')
        f.write("{}".format(choice_limit(read_file("limit_3.txt")[0])))
        f.close()
    if len(read_file("limit_4.txt")) == 0:
        f = open("limit_4.txt", 'w')
        f.write("{}".format(choice_limit(read_file("limit_4.txt")[0])))
        f.close()
    if len(read_file("limit_5.txt")) == 0:
        f = open("limit_5.txt", 'w')
        f.write("{}".format(choice_limit(read_file("limit_5.txt")[0])))
        f.close()


def final():
    global direct, data_file, matrix, data_limit_1
    update_data(direct, data_file)
    res = fff(matrix)
    tp1 = res[0]
    tp2 = res[1]
    value = res[2]

    plot_hist(tp1, tp2, value)
    result_arr = []
    max_index = value.index(max(value))
    result = int(tp1[max_index]) * int(tp2[max_index])
    result_arr.append(result)

    medium_index = value.index(list(set(sorted(value)))[int(len(set(value)) / 2)])
    result2 = int(tp1[medium_index]) * int(tp2[medium_index])
    result_arr.append(result2)

    result3 = int(max(tp1)) * int(max(tp2))
    result_arr.append(result3)

    diagonal = []
    medium_elem = list(set(sorted(tp1)))[int(len(set(tp1)) / 2)]
    for el in range(len(tp1)):
        if tp1[el] >= medium_elem:
            diagonal.append(value[el])
    result4 = (max(diagonal))
    result_arr.append(result4)

    medium_scale_index = tp1.index(list(set(sorted(tp1)))[int(len(set(tp1)) / 2)])
    result5 = value[medium_scale_index]
    result_arr.append(result5)

    color = []
    y_value_n = []
    y_value = []
    max_sc_arr = []
    for data_file, limit_file in [['data_1.txt', 'limit_1.txt'], ['data_2.txt', 'limit_2.txt'],
                                  ['data_3.txt', 'limit_3.txt'], ['data_4.txt', 'limit_4.txt'],
                                  ['data_5.txt', 'limit_5.txt']]:
        with open(data_file) as file:
            scale_array = []
            value_scale = [row.strip() for row in file]

        for value in value_scale:
            scale_array.append([int(value.split(" ")[0]), int(value.split(" ")[1])])
        max_sc = max(np.array(scale_array)[:, 0])
        y_value.append(int(read_file(limit_file)[0]) * 100 / max_sc)
        max_sc_arr.append(max_sc)

        # if int(result) > int(check_data(direct, limit_file)[0][0]):
        #     color.append('r')
        # else:
        #     color.append('g')
    for i in range(5):
        y_value_n.append(result_arr[i] * 100 / max_sc_arr[i])

    y_value_n = tuple(y_value_n[:3])
    y_value = tuple(y_value[:3])

    ind = np.arange(len(y_value_n))  # the x locations for the groups
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, y_value_n, width,
                    color='green', label='Norma')
    rects2 = ax.bar(ind + width / 2, y_value, width,
                    color='Red', label='Image')

    ax.set_ylabel('%')
    ax.set_xticks(ind)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xticklabels(('1', '2', '3'))
    ax.legend()

    # plt.show()

    plt.savefig('out_fig.png')
    foo = Image.open("out_fig.png")
    foo = foo.resize((200, 200), Image.ANTIALIAS)
    foo.save("out_fig.png", quality=95)
    print(1)
    # plt.show()


# I downsize the image with an ANTIALIAS filter (gives the highest quality)


def plot_hist(tp1, tp2, value):
    cmap = cm.get_cmap('jet')
    max_height = np.max(value)
    min_height = np.min(value)
    rgba = [cmap((k - min_height) / max_height) for k in value]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(0, 0, 0, tp1, tp2, value, color=rgba)


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


# def check_3():
#     for i in range(50:70)

if __name__ == "__main__":
    pygame.init()
    Tk().withdraw()
    filename = askopenfilename(filetypes=(("jpg file", "*.jpg"), ("bmp file", "*.bmp"), ("All Files", "*.*")))
    direct = 'Image'
    data_file = "data_1.txt"
    data_limit_1 = "limit_1.txt"

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
    # pygame.display.quit()
    im.save(output_loc)
    img_rgb = cv2.imread('out.bmp')
    # Convert it to grayscale
    matrix = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    final()


    # setup("out_fig.png")

    class GameObj:
        def __init__(self, img, x, y, step):
            self.img = img  # Картинка объекта
            self.x = x  # x, y - коодинаты начального положения
            self.y = y
            self.pos = img.get_rect().move(x, y)


    avatar = pygame.image.load('out_fig.png')

    # Инициируем игровой объект
    x = GameObj(avatar, 0, 250, 10)

    # Рисуем картинку объекта, в его координатах
    screen.blit(x.img, x.pos)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
