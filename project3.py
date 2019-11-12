import fft
import matplotlib.pyplot as plt
import numpy
from math import cos, pi, floor, sqrt, log
from PIL import Image

fft_func = fft.FFT()

experiment1 = False
experiment2 = True
experiment3 = True

# Experiment 1
# Part 1
def center_transform(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i]*(-1)**(i))
    return new_data

def plot_transform_1d(data, centered=True):
    if centered:
        x = range(-len(data)//4, len(data)//4)
    else:
        x = range(len(data)//2)
    mag = [(data[2*i]**2 + data[2*i+1]**2)**(1/2) for i in range(len(data)//2)]
    plt.plot(x, [data[2*i] for i in range(len(data)//2)], label="real")
    plt.plot(x, [data[2*i+1] for i in range(len(data)//2)], label="imaginary")
    plt.plot(x, mag, label="magnitude")
    plt.axhline(0, lw=1, color="grey")
    plt.legend()
    plt.show()

def gen_cos_wave(u, N):
    return [cos(2*pi*u*x/N) for x in range(N)]

if experiment1:
    data = [2, 3, 4, 4]
    transform = fft_func.fourier_transform(data, 1)
    plot_transform_1d(transform, centered=False)
    print(transform)

    inverse_transform = fft_func.fourier_transform(transform, -1)
    print(inverse_transform)

    # Part 2
    cos_wave = gen_cos_wave(8, 128)
    plt.plot(cos_wave)
    plt.show()
    cos_transform = fft_func.fourier_transform(center_transform(cos_wave), 1)
    plot_transform_1d(cos_transform)

    # Part 3
    rect_func = []
    with open("Rect_128.dat") as rect_file:
        for line in rect_file:
            rect_func.append(float(line))
    plt.plot(rect_func)
    plt.show()

    rect_transform = fft_func.fourier_transform(center_transform(rect_func), 1)
    plot_transform_1d(rect_transform)

# Experiment 2
#uses Experiment 1's DFT and just goes across the rows
def dft2d(data, mode):

    transform = [[0 for x in range(len(data[0])*2)] for y in range(len(data[0])*2)] 
    for x in range(0, len(data[0])):
        transform[x] = fft_func.fourier_transform(data[x], mode)
        if(mode == -1):
            for y in range(0, len(data[0])):
                multiplier = 10 ** 3
                transform[x][y] = floor(transform[x][y]*multiplier + 0.5) / multiplier
    return transform

#make a single square of size squareSize with a black background
def SquareImage(squareSize):
    size = squareSize/2
    size = int(size)
    im = Image.new("L", (512, 512), "black")
    im.paste("white", (256-size,256-size, 256+size,256+size))
    return im

def frequencyMagnitude(transform):
    magnitude = [[0 for x in range(len(transform[0]))] for y in range(len(transform[0]))]
    for x in range(0, len(transform[0])):
        for y in range(0, len(transform[0]), 2):
            magnitude[x][y] = sqrt((transform[x][y]**2) + (transform[x][y+1]**2))
    return magnitude

def inverseNoZeros(OG, FFT):
    realImage = [[0 for x in range(len(OG[0]))] for y in range(len(OG[0]))]
    realImageX = 0
    realImageY = 0
    for x in range(len(FFT[0])):
        for y in range(len(FFT[0])):
            if FFT[x][y] != 0.0:
                realImage[realImageX][realImageY] = FFT[x][y]
                realImageY += 1
                if realImageY >= len(OG[0]):
                    realImageX += 1
                    realImageY = 0
    return realImage

def logarithmicScaling(realImage):
    largestValue = 0
    for x in range(len(realImage[0])):
        for y in range(len(realImage[0])):
            if largestValue < realImage[x][y]:
                largestValue = realImage[x][y]

    logRealValues = [[0 for x in range(len(realImage[0]))] for y in range(len(realImage[0]))]
    c = 255/log(1 + largestValue)
    
    for x in range(len(realImage[0])):
        for y in range(len(realImage[0])):
            logRealValues[x][y] = c*log(1 + abs(realImage[x][y]))
    return logRealValues

def frequencyTranslation(realImage):
    translatedVals = [[0 for x in range(len(realImage[0]))] for y in range(len(realImage[0]))]
    for x in range(len(realImage[0])):
        for y in range(len(realImage[0])):
            translatedVals[x][y] = translatedVals[x][y] * ((-1)**(x+y))
    
    return translatedVals

def testfrequencyTranslation(realImage):
    translatedVals = [[0 for x in range(len(realImage[0]))] for y in range(len(realImage[0]))]
    for x in range(len(realImage[0])):
        for y in range(len(realImage[0])):
            translatedVals[x][y] = realImage[(len(realImage[0])-x-1)][(len(realImage[0])-y-1)]

    return translatedVals

# Part1
if experiment2:
    #convert from square image
    squareImage = SquareImage(32)
    squareImage.save("square32.pgm", "PPM")
    numpyArray = numpy.array(squareImage)
    #to numpy array
    squareList = numpyArray.tolist()

    # run 2dFFT where 1 is forward and -1 is inverse FFTs
    transform = dft2d(squareList, 1)
    translatedImage = testfrequencyTranslation(transform)
    magnitude = frequencyMagnitude(translatedImage)

   # print(magnitude[256])
    magnitudeArray = numpy.asarray(magnitude)
    
    magnitudeImage = Image.fromarray(magnitudeArray)
    inverseTransform = dft2d(magnitude, -1)
   # print(inverseTransform[256])
    realImage = inverseNoZeros(squareList, inverseTransform)

    inverseArray = numpy.asarray(realImage)
    inverseImage = Image.fromarray(inverseArray)
    inverseImage.show()
    logRealValues = logarithmicScaling(inverseArray)
 
    logArray = numpy.asarray(logRealValues)
    logImage = Image.fromarray(logArray)
    logImage.show()
    
# Part2
# Experiment 3
