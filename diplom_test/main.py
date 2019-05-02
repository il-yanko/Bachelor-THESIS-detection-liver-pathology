# outer dependencies:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import os.path
import scipy.stats as stats

# inner dependencies:
from glcm import GLCM
from pe import PE
from img_reader import IMGReader
import gradients as grds


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
# normalization to the summ
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

#print("Donwloading images for a pathological and a normal state of a kidney parenchyma for our ultrasonography:")
pathoNames = ["auh","dsh","gpb","gpc","vls"]

# getting the current working directory
cwd = os.getcwd()

norma = IMGReader.read_directory(cwd + "/data/general/norma/")
patho = IMGReader.read_directory(cwd + "/data/general/ne-norma/")
auh = IMGReader.read_directory(cwd + "/data/" + pathoNames[0] + "/")
dsh = IMGReader.read_directory(cwd + "/data/" + pathoNames[1] + "/")
gpb = IMGReader.read_directory(cwd + "/data/" + pathoNames[2] + "/")
gpc = IMGReader.read_directory(cwd + "/data/" + pathoNames[3] + "/")
vls = IMGReader.read_directory(cwd + "/data/" + pathoNames[4] + "/")
array = []
array.append(auh)
array.append(dsh)
array.append(gpb)
array.append(gpc)
array.append(vls)
# TODO: make some data structure for all that


# read 1 image as a test
'''
original = mpimg.imread("pathology/14.bmp")
original = original[0:5,0:5,:]
rawGray = rgb_to_gray(original)
gray = np.array(rawGray)
#plt.imshow(original)
#plt.imsave("tmp/tmp1.png", original)
#calculation = glcm.GLCM(gray).glcm_complex_duplex()
## GLCM results' saving
#plt.imshow(calculation)
'''

# save all 50 GLCMs as temporary
'''
for i in range(len(normaBMP)):
    curIm = normaBMP[i]
    calculation = GLCM(curIm).glcm_complex_duplex()
    number = i+1
    print(number)
    plt.imshow(calculation)
    path = "glcm/n/n" + str(number) + ".png"
    plt.imsave(path, calculation, cmap="inferno")
print("norma was saved successfully")
for i in range(len(pathoBMP)):
    curIm = pathoBMP[i]
    calculation = GLCM(curIm).glcm_complex_duplex()
    number = i+1
    print(number)
    plt.imshow(calculation)
    path = "glcm/p/p" + str(number) + ".png"
    plt.imsave(path, calculation, cmap="inferno")
print("pathology was saved successfully")
'''

numberParam = len(param)
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

# build histograms for different diseases comparing with norma

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
    plt.subplot(number+1,1,i+1)
    histogram_average_gray_frequency(norma,'norma','b',0.7)
    histogram_average_gray_frequency(array[i],pathoNames[i],'r',0.9)
    plt.title(pathoNames[i] + ' + norma')
    plt.ylabel('% samples') #, fontsize=18
    plt.xlabel('color')
    plt.xlim(0,255)
    plt.legend(loc='best')
plt.subplot(number+1,1,number+1)
histogram_average_gray_frequency(norma,'norma','b',0.7)
histogram_average_gray_frequency(patho,'patho','r',0.9)
plt.title('patho' + ' + norma')
plt.ylabel('% samples') #, fontsize=18
plt.xlabel('color')
plt.xlim(0,255)
plt.legend(loc='best')
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

