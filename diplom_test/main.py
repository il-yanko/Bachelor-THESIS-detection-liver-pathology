import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import os, os.path
import glcm
import pe
import gradients as grds
from PIL import Image
import math

#download folder BMP
def get_all_BMP(fullDir):
    # считаем количество изображений
    fileNumber = len(next(os.walk(fullDir))[2])
    print(fileNumber, "images were found")
    imgArr = []
    for i in range(1, fileNumber + 1):
        imgArr.append(mpimg.imread(fullDir + '/' + str(i) + ".bmp"))
    print(len(imgArr), "images were downloaded")
    # TODO: дописать исключения
    return imgArr
#process RGB/grayscale
def rgb_to_gray(rgb):
    #скалярное произведение начальных цветов с определенными теоретическими коэффициентами по системе YUV
    return np.dot(rgb[...,:3], [0.333, 0.333, 0.333]).round(3).astype(int)
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
#normalization to the summ
def normalize_array_2D(table):
    # to divide each cell by the sum of all cells
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
# getting the current working directory
cwd = os.getcwd()

print("Donwloading images for a pathological and a normal state of a kidney parenchyma for our ultrasonography:")
def get_all_img_make_gray(cwd, folderName):
    path = cwd + "/" + folderName
    print("\nPath = ", path)
    imgArray = get_all_BMP(path)
    for i in range(len(imgArray)):
        imgArray[i] = rgb_to_gray(imgArray[i])
    return imgArray
#patho = get_all_img_make_gray(cwd, "pathology")
norma = get_all_img_make_gray(cwd, "norma")

pathoNames = ["auh","dsh","gpb","gpc","vls"]
# TODO: сделать какую-то структуру (может, деку или что-то такое)
auh = get_all_img_make_gray(cwd, "data/" + pathoNames[0])
dsh = get_all_img_make_gray(cwd, "data/" + pathoNames[1])
gpb = get_all_img_make_gray(cwd, "data/" + pathoNames[2])
gpc = get_all_img_make_gray(cwd, "data/" + pathoNames[3])
vls = get_all_img_make_gray(cwd, "data/" + pathoNames[4])
array = []
array.append(auh)
array.append(dsh)
array.append(gpb)
array.append(gpc)
array.append(vls)


## считывание 1го изображение для тестов
#original = mpimg.imread("pathology/14.bmp")
##original = original[0:20,0:20,:]
#fig = plt.figure()
##plt.subplot(211)
##plt.imshow(original)
##plt.title("Оригінальне зображення")
##plt.imsave("tmp/tmp1.png", original)
#rawGray = rgb2gray(original)
#gray = np.array(rawGray)
##calculation = glcm.GLCM(gray).glcm_complex_duplex()
##grCoMap = plt.get_cmap('gray')
#ax = fig.add_subplot(111)
## GLCM results' saving
##plt.imshow(calculation)
##cmap=grCoMap
#grFr = greyFrequencies(gray)
##grHist = plt.hist(gray)
##print(grHist)
#plt.bar(np.arange(0,255,1),grFr[0], color='g', alpha=0.1)


# build histograms for different diseases comparing with norma
'''
def greyFrequencies(img):
    size1 = len(img)
    size2 = len(img[0])
    #print('s1=',size1,' s2=',size2)
    rez = np.zeros((1,255))
    for i in range (size1):
        for j in range (size2):
            rez [0][ img[i][j] ] += 1
    return rez
def histogram_average_gray_frequency(imgAr,name,color='b',alpha=1.):
    rez = np.zeros((1, 255))
    for i in range(len(imgAr)):
        gray = np.array(imgAr[i])
        grFr = greyFrequencies(gray)
        relative = np.zeros((1,255))
        summ = np.sum(grFr[0])
        for j in range (len(grFr[0])):
            relative[0][j] = grFr[0][j] * 100 / summ

        for j in range(len(rez[0])):
            rez[0][j] += relative[0][j]
    for k in range(len(rez[0])):
        rez[0][k] /= len(imgAr[0])
    plt.bar(np.arange(0,255,1),rez[0],color=color,alpha=alpha,label=name)
number = len(pathoNames)
for i in range (number):
    plt.subplot(number,1,i+1)
    histogram_average_gray_frequency(norma,'norma','b',0.7)
    histogram_average_gray_frequency(array[i],pathoNames[i],'r',0.9)
    plt.title(pathoNames[i] + ' + norma')
    plt.ylabel('% samples') #, fontsize=18
    plt.xlabel('color')
    plt.xlim(0,255)
    plt.legend(loc='best')
'''

# attempt to plot the red-blue mask on the gray image
'''
#img = Image.fromarray(grayAsRGB.astype(np.uint8))
img = Image.open("tmp/tmp2.png")
plt.subplot(312)
plt.imshow(np.array(img))
plt.title("Аугментоване зображення")
px = img.load()
maxFreq = np.amax(calculation)
maxRed  = 0.5
maxBlue = 0.3
reds  = np.arange(0,maxRed, 0.1) * 255
blues = np.arange(0,maxBlue,0.1) * 255
def paintPixelPIL(img,x,y,r,g,b):
    R = img.getpixel((x, y))[0] + r
    G = img.getpixel((x, y))[1] + g
    B = img.getpixel((x, y))[2] + b
    img.putpixel((x, y), (R, G, B))
    return img
print('len(ed)-1=',len(ed)-1,'\n')
print('len(ed[0])-1',len(ed[0])-1,'\n')
for i in range (len(ed)-1):
    for j in range (len(ed[0])-1):
        if calculation [ed[i][j]-1] [ed[i][j+1]-1] <= 0.15*maxFreq:
            paintPixelPIL(img, i, j,   90, 0, 0)
            paintPixelPIL(img, i, j+1, 90, 0, 0)
        else:
            if calculation [ed[i][j]-1] [ed[i][j+1]-1] <= 0.3*maxFreq:
                paintPixelPIL(img, i, j,   55, 0, 0)
                paintPixelPIL(img, i, j+1, 55, 0, 0)
            else:
                if calculation[ed[i][j] - 1][ed[i][j + 1] - 1] <= 0.45*maxFreq:
                    paintPixelPIL(img, i, j,     40, 0, 40)
                    paintPixelPIL(img, i, j + 1, 40, 0, 40)
                else:
                    if calculation[ed[i][j] - 1][ed[i][j + 1] - 1] <= 0.6 * maxFreq:
                        paintPixelPIL(img, i, j, 0, 0, 55)
                        paintPixelPIL(img, i, j + 1, 0, 0, 55)
                    else:
                        if calculation[ed[i][j] - 1][ed[i][j + 1] - 1] <= 0.75 * maxFreq:
                            paintPixelPIL(img, i, j, 0, 0, 90)
                            paintPixelPIL(img, i, j + 1, 0, 0, 90)

img.save("tmp/colorized.png", "PNG", quality = 100)
im2arr = np.array(img)
plt.subplot(313)
plt.imshow(im2arr)
plt.title("Зображення, покрите маскою на основі GLCM")
'''

#plt.show()
