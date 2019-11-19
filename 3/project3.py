import fft
from PIL import Image
import matplotlib.pyplot as plt
from math import sin, cos, pi, atan2, log, sqrt
import numpy as np

fft_func = fft.FFT()

experiment1 = False
experiment2 = False
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


def add_zero(data):
    new_data = []
    for x in range(len(data)):
        new_data.append([])
        for y in range(len(data[x])):
            new_data[x].append(data[x][y])
            new_data[x].append(0)
    return new_data

def dft2d(data, mode):
    n = len(data[0])
    if mode is 1:
        data = add_zero(data)

    transform = []
    # dft of rows
    for i in range(0, len(data)):
        transform.append(fft_func.fourier_transform(data[i], mode, extend=False))

    # multiply by n
    if mode is 1:
        for i in range(len(transform)):
            for j in range(len(transform[i])):
                transform[i][j] *= n
    # get columns
    columns = []
    for i in range(len(transform[0]) // 2):
        columns.append([])
        for row in transform:
            columns[i].append(row[i*2])
            columns[i].append(row[i*2+1])
    # transform columns
    for x in range(0, len(columns)):
        columns[x] = fft_func.fourier_transform(columns[x], mode, extend=False)
    # convert columns to rows
    # get rows
    transform = []
    for i in range(len(columns[0]) // 2):
        transform.append([])
        for row in columns:
            transform[i].append(row[i * 2])
            transform[i].append(row[i * 2 + 1])

    return transform

def normalize_0_255(input_list):
    min = 999999
    max = -999999
    for width in range(len(input_list)):
        for height in range(len(input_list[width])):
            if min > input_list[width][height]:
                min = input_list[width][height]
            if max < input_list[width][height]:
                max = input_list[width][height]

    for width in range(len(input_list)):
        for height in range(len(input_list[width])):
            input_list[width][height] = int((input_list[width][height] - min) * (255 / (max - min)))

    return input_list

def center_2d_transform(data):
    new_data = []
    for i in range(len(data)):
        new_data.append([])
        for j in range(len(data[i])):
            new_data[i].append(data[i][j]*(-1)**(i+j))
    return new_data

def remove_centering_2d_transform(data):
    data = remove_zeros(data)
    new_data = []
    for i in range(len(data)):
        new_data.append([])
        for j in range(len(data[i])):
            new_data[i].append(data[i][j] * (-1) ** (i + j))
    return new_data

def remove_zeros(data):
    new_data = []
    for i in range(len(data)):
        new_data.append([])
        for j in range(0, len(data[i]), 2):
            new_data[i].append(data[i][j])
    return new_data

def plot_transform_2d(data, centered=True):
    if centered:
        x = range(-(len(data) * 2) // 4, (len(data) * 2) // 4)
        y = range(-len(data[0]) // 4, len(data[0]) // 4)
    else:
        x = range((len(data) * 2) // 2)
        y = range(len(data[0]) // 2)

    # magnitude
    mag = []
    for i in range(len(data)):
        mag.append([])
        for j in range(len(data[i]) // 2):
            mag[i].append(log(((data[i][2 * j] ** 2) + (data[i][2 * j + 1] ** 2)) ** (1 / 2) + 1))
        print()

    plt.imshow(mag, label="magnitude", cmap=plt.get_cmap("gray"))
    plt.show()

def SquareImage(squareSize):
    size = squareSize/2
    size = int(size)
    im = Image.new("L", (512, 512), "black")
    im.paste("white", (128-size,128-size, 128+size,128+size))
    im.paste("white", (256-size, 256-size, 256+size, 256+size))
    return im

if experiment2:
    #convert from square image
    squareImage = SquareImage(128)
    numpyArray = np.array(squareImage)
    #to numpy array
    squareList = numpyArray.tolist()

    # run 2dFFT where 1 is forward and -1 is inverse FFTs
    plt.imshow(squareList, cmap=plt.get_cmap("gray"))
    plt.show()
    transform = dft2d(center_2d_transform(squareList), 1)

    plot_transform_2d(transform)
    # Set phase equal to zero
    for i in range(len(transform)):
        for j in range(0, len(transform[i]), 2):
            transform[i][j] = sqrt((transform[i][j] ** 2) + (transform[i][j + 1] ** 2))
            transform[i][j + 1] = 0

    inverseTransform = dft2d(transform, -1)
    inverseTransform = remove_centering_2d_transform(inverseTransform)
    plt.imshow(inverseTransform, cmap=plt.get_cmap("gray"))
    plt.show()
    print()

# Experiment 3

if experiment3:
    lenna = Image.open("../images-pgm/lenna.pgm")
    width, height = lenna.size
    lenna_pixels = lenna.load()

    lenna_list = []
    for x in range(width):
        lenna_list.append([])
        for y in range(height):
            lenna_list[x].append(lenna.getpixel((y,x)))

    # test
    plt.imshow(lenna_list, cmap=plt.get_cmap("gray"))
    plt.show()
    lenna_transform = dft2d(center_2d_transform(lenna_list), 1)
    plot_transform_2d(lenna_transform)
    lenna_inverse_transform = dft2d(lenna_transform, -1)
    lenna_inverse_transform = remove_centering_2d_transform(lenna_inverse_transform)
    plt.imshow(lenna_inverse_transform, cmap=plt.get_cmap("gray"))
    plt.show()


    # part 3a
    lenna_transform = dft2d(center_2d_transform(lenna_list), 1)


    plot_transform_2d(lenna_transform)

    # Set phase equal to zero
    for i in range(len(lenna_transform)):
        for j in range(0, len(lenna_transform[i]), 2):
            lenna_transform[i][j] = sqrt((lenna_transform[i][j] ** 2) + (lenna_transform[i][j + 1] ** 2))
            lenna_transform[i][j+1] = 0

    lenna_transform = dft2d(lenna_transform, -1)
    lenna_transform = remove_centering_2d_transform(lenna_transform)
    # lenna_inverse_transform = remove_zeros(lenna_inverse_transform)
    plt.imshow(lenna_transform, cmap=plt.get_cmap("gray"))
    plt.show()

    # part 3b
    lenna_transform = dft2d(center_2d_transform(lenna_list), 1)
    plot_transform_2d(lenna_transform)

    # Set phase to original and magnitude to 1
    for i in range(len(lenna_transform)):
        for j in range(0, len(lenna_transform[i]), 2):
            theta = (atan2(lenna_transform[i][j+1], lenna_transform[i][j]))
            lenna_transform[i][j] = cos(theta)
            lenna_transform[i][j+1] = sin(theta)

    plot_transform_2d(lenna_transform)

    lenna_inverse_transform = dft2d(lenna_transform, -1)
    lenna_inverse_transform = remove_centering_2d_transform(lenna_inverse_transform)
    lenna_inverse_transform = normalize_0_255(lenna_inverse_transform)
    plt.imshow(lenna_inverse_transform, cmap=plt.get_cmap("gray"))
    plt.show()
    print()

# data = [[1, 2, 3, 4, 5],[249, 250, 251, 253, 254],[1, 2, 3, 4, 5],[249, 250, 251, 253, 254],[50,50,50,50,50]]
# plt.imshow(data, cmap=plt.get_cmap("gray"))
# plt.show()
# transform = dft2d(center_2d_transform(data), 1)
# plot_transform_2d(transform, centered=True)
# inverse_transform = dft2d(transform, -1)
# inverse_transform = remove_centering_2d_transform(inverse_transform)
# plt.imshow(inverse_transform, cmap=plt.get_cmap("gray"))
# plt.show()
# print()