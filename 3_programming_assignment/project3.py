import fft
import matplotlib.pyplot as plt
import numpy
from math import cos, pi, floor, sqrt, log
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

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
    transform = []
    if mode is 1:
        for y in range(0, len(data[0])):
            transform.append(fft_func.fourier_transform(data[y], mode))
    else:
        for y in range(0, len(data)):
            transform.append(fft_func.fourier_transform(data[y], mode))

    return transform


#make a single square of size squareSize with a black background
def SquareImage(squareSize):
    size = squareSize/2
    size = int(size)
    im = Image.new("L", (512, 512), "black")
    im.paste("white", (256-size,256-size, 256+size,256+size))
    return im

def normalize_0_255(input_list):
    returnList = []
    min = 999999
    max = -999999
    for width in range(len(input_list)):
        for height in range(len(input_list[width])):
            if min > input_list[width][height]:
                min = input_list[width][height]
            if max < input_list[width][height]:
                max = input_list[width][height]

    for width in range(len(input_list)):
        returnList.append([])
        for height in range(len(input_list[width])):
            returnList[width].append(int((input_list[width][height] - min) * (255 / (max - min))))

    return returnList

def center_2d_transform(data):
    new_data = []
    for i in range(len(data)):
        new_data.append([])
        for j in range(len(data[i])):
            new_data[i].append(data[i][j]*(-1)**(i+j))
    return new_data

def plot_transform_2d(data, centered=False):

    mag = []
    for i in range(len(data)//2):
        mag.append([])
        for j in range(len(data[i])//2):
            mag[i].append((data[2*i][2*j]**2 + data[2*i+1][2*j+1]**2)**(1/2))
    fig = plt.figure()

    plt.imshow(mag, label="magnitude")
    plt.legend()
    plt.show()

def squareExperiment(squareSize):
    #convert from square image
    squareImage = SquareImage(squareSize)
    """squareImage.show()
                string = "Square"
                string += str(squareSize)
                string += ".pgm"
                squareImage.save(string)"""
    numpyArray = numpy.array(squareImage)
    #to numpy array
    squareList = numpyArray.tolist()

    # run 2dFFT where 1 is forward and -1 is inverse FFTs
    transform = numpy.fft.fft2(squareList)

    magnitude = normalize_0_255(transform)

    plot_transform_2d(magnitude)

    centered = numpy.fft.fftshift(magnitude)

    plot_transform_2d(centered)

# Part1
if experiment2:
    squareExperiment(32)
    squareExperiment(64)
    squareExperiment(128)

# Part2
# Experiment 3
