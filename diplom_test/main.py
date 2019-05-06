# outer dependencies:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import os.path
import time

# inner dependencies:
from glcm import GLCM, calculate_save_glcm
from img_reader import IMGReader, rgb_to_gray
import processing as proc

#=========================================

print("Donwloading images for a pathological and a normal state of a kidney parenchyma for our ultrasonography:")
# getting the current working directory
cwd = os.getcwd()
pathoNames = ["auh", "dsh", "gpb", "gpc", "vls"]
norma = IMGReader.read_directory(cwd + "/data/general/norma/")
patho = IMGReader.read_directory(cwd + "/data/general/ne-norma/")
auh = IMGReader.read_directory(cwd + "/data/" + pathoNames[0] + "/")
dsh = IMGReader.read_directory(cwd + "/data/" + pathoNames[1] + "/")
gpb = IMGReader.read_directory(cwd + "/data/" + pathoNames[2] + "/")
gpc = IMGReader.read_directory(cwd + "/data/" + pathoNames[3] + "/")
vls = IMGReader.read_directory(cwd + "/data/" + pathoNames[4] + "/")
array = list()
array.append(auh)
array.append(dsh)
array.append(gpb)
array.append(gpc)
array.append(vls)
# TODO: make some data structure for all that

# read 1 image as a test
'''
original = mpimg.imread("pathology/14.bmp")
original = original[0:3,0:3,]
gray = np.array(rgb_to_gray(original))
print("gray = \n%s"%gray)
a = []
a.append(gray)
#plt.imshow(original)
#plt.imsave("tmp/tmp1.png", original)
#calculation = glcm.GLCM(gray).glcm_complex_duplex()
## GLCM results' saving
#plt.imshow(calculation)
'''


# processing and saving all GLCM-s (.csv and .png)
'''
start_time = time.time()
calculate_save_glcm("auh", auh)
calculate_save_glcm("dsh", dsh)
calculate_save_glcm("gpb", gpb)
calculate_save_glcm("gpc", gpc)
calculate_save_glcm("vls", vls)
calculate_save_glcm("n", norma)
calculate_save_glcm("p", patho)
minutes = (time.time() - start_time) / 60
print("--- %s minutes have passed ---" % minutes)
'''


numberParam = proc.paramNumber
numberPatho = len(pathoNames)
# build scatters for different 1st order stats of diseases comparing with norma
'''
def pseudo_scatter(fig,imgAr,name,color='b',alpha=1.,marker="."):
    global param
    try:
        for i in range(len(imgAr)-1):
            gray = np.array(imgAr[i])
            stat = calculate_first_order_statistic_2D(gray)
            for j in range(numberParam):
                ax = fig.add_subplot(numberParam,2,j+1)
                ax.set_title(param[j])
                ax.scatter(stat[param[j]],stat[param[j]],color=color,
                           alpha=alpha,marker=marker)
        gray = np.array(imgAr[len(imgAr)-1])
        stat = calculate_first_order_statistic_2D(gray)
        for j in range(numberParam):
            ax = fig.add_subplot(numberParam, 2, j + 1)
            ax.set_title(param[j])
            ax.scatter(stat[param[numberParam-1]], stat[param[numberParam-1]], color=color,
                   alpha=alpha, label=name, marker=marker)
            ax.legend(loc='best')
    except IndexError:
        print("Wrong index was chosen!")

for i in range (numberPatho):
    data = array[i]
    fig = plt.figure(num=pathoNames[i]+' + norma',figsize=(10, 10))
    pseudo_scatter(fig, norma,'norma','b',0.6,"v")
    pseudo_scatter(fig, array[i],pathoNames[i],'r',0.6,"^")
    fig.tight_layout()
plt.show()
'''


# build color distribution for different diseases comparing with norma
'''
number = len(pathoNames)
for i in range(number):
    plt.subplot(number+1, 1, i+1)
    proc.average_gray_frequency_distribution(plt, norma, 'norma', 'b', 0.6)
    proc.average_gray_frequency_distribution(plt, array[i], pathoNames[i], 'r', 0.6)
    plt.title(pathoNames[i] + ' + norma')
    plt.ylabel('% samples')
    plt.xlabel('color')
    plt.xlim(0, 255)
    plt.ylim(0, 4.5)
    plt.legend(loc='best')
plt.subplot(number+1, 1, number+1)
proc.average_gray_frequency_distribution(plt, norma, 'norma', 'b', 0.6)
proc.average_gray_frequency_distribution(plt, patho, 'patho', 'r', 0.6)
plt.title('patho + norma')
plt.ylabel('% samples')
plt.xlabel('color')
plt.xlim(0, 255)
plt.ylim(0, 4.5)
plt.legend(loc='best')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
'''

# build color distribution for different diseases comparing with each other
proc.average_gray_frequency_distribution(plt, array[0], pathoNames[0], 'red', 0.9)
proc.average_gray_frequency_distribution(plt, array[1], pathoNames[1], 'black', 0.9)
proc.average_gray_frequency_distribution(plt, array[2], pathoNames[2], 'gold', 0.9)
proc.average_gray_frequency_distribution(plt, array[3], pathoNames[3], 'olive', 0.9)
proc.average_gray_frequency_distribution(plt, array[4], pathoNames[4], 'blue', 0.9)

plt.title('patho + norma')
plt.ylabel('% samples')
plt.xlabel('color')
plt.xlim(0, 255)
plt.ylim(0, 4.5)
plt.legend(loc='best')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()


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


# binarization of the image
'''
photo = norma[0]
plt.subplot(2,1,1)
plt.imshow(photo,cmap="gray")
# change!
threshold = 100
photo = np.where(photo>threshold,255,0)
plt.subplot(2,1,2)
plt.imshow(photo,cmap="gray")
plt.show()
'''