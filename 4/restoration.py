from PIL import Image
import fft
import matplotlib.pyplot as plt
from math import cos, sin, pi, e, log
import numpy as np

def add_zero(data):
    new_data = []
    for x in range(len(data)):
        new_data.append([])
        for y in range(len(data[x])):
            new_data[x].append(data[x][y])
            new_data[x].append(0)
    return new_data

def remove_zeros(data):
    new_data = []
    for i in range(len(data)):
        new_data.append([])
        for j in range(0, len(data[i]), 2):
            new_data[i].append(data[i][j])
    return new_data

def plot_transform_2d(data, use_log=True):
    # magnitude
    mag = []
    for i in range(len(data)):
        mag.append([])
        for j in range(len(data[i]) // 2):
            if use_log:
                mag[i].append(log(((data[i][2*j]**2) + (data[i][2*j + 1]**2))**(1/2) + 1))
            else:
                mag[i].append(((data[i][2*j]**2) + (data[i][2*j + 1]**2))**(1/2))

    plt.imshow(mag, label="magnitude", cmap=plt.get_cmap("gray"))
    plt.show()

def dft2d(data, mode):
    n = len(data)
    if mode is 1:
        data = add_zero(data)

    transform = []
    # dft of rows
    for i in range(len(data)):
        transform.append(fft_func.fourier_transform(data[i], mode, extend=False))

    # multiply by n
    if mode is 1:
        for i in range(len(transform)):
            for j in range(len(transform[i])):
                transform[i][j] *= n

    # get columns
    columns = []
    for x in range(len(transform)):
        columns.append([])
        for y in range(len(transform)):
            columns[x].append(transform[y][x*2])
            columns[x].append(transform[y][x*2+1])

    # transform columns
    for x in range(0, len(columns)):
        columns[x] = fft_func.fourier_transform(columns[x], mode, extend=False)
    # convert columns to rows
    transform = []
    for x in range(len(columns)):
        transform.append([])
        for y in range(len(columns)):
            transform[x].append(columns[y][x*2])
            transform[x].append(columns[y][x*2+1])

    return transform

def center_2d_transform(data):
    new_data = []
    for i in range(len(data)):
        new_data.append([])
        for j in range(len(data[i])):
            new_data[i].append(data[i][j]*(-1)**(i+j))
    return new_data

def remove_centering_2d_transform(data):
    data = remove_zeros(data)
    return center_2d_transform(data)

def motion_blur():
    pass

fft_func = fft.FFT()

def matmul(a, b):
    result = []
    for i in range(len(a)):
        result.append([])
        for j in range(len(b[0])):
            sum = 0
            for k in range(len(b)):
                sum += a[i][k]*b[k][j]
            result[-1].append(sum)

    return result

# Hadamard is just broadcasting (element by element mlutiplication
def complex_hadamard(a, b):
    result = []
    for i in range(len(a)):
        result.append([])
        for j in range(len(a[0])//2):
            real_sum = a[i][2*j]*b[i][2*j] - a[i][2*j+1]*b[i][2*j+1]
            complex_sum = a[i][2*j]*b[i][2*j+1] + a[i][2*j+1]*b[i][2*j]
            result[-1].append(real_sum)
            result[-1].append(complex_sum)

    return result

def pad_zeros(image, filter):
    n = len(image)
    m = len(filter)

    # Padding must be at least n+m-1
    min_padding = n+m-1
    # Pad to a power of 2
    padding = 0
    exponent = 1
    while padding < min_padding:
        padding = 2**exponent
        exponent+=1

    for i in range(m,padding):
        filter.append([0]*(padding))
    for i in range(m):
        filter[i].extend([0]*(padding-m))

    for i in range(n,padding):
        image.append([0]*(padding))
    for i in range(n):
        image[i].extend([0]*(padding-n))

    return image, filter, padding - n

def unpad(image, pad_len):
    for i in range(pad_len):
        image[i] = image[i][:pad_len*2]

    return image[:pad_len]

a = [[1, 1, 3, 0],
     [2, 0, 1, 1]]

b = [[2,  0, 1, 0],
     [2, -1, 3, 3]]

print(complex_hadamard(a, b))

image = Image.open("nate.pgm")

width, height = image.size
image_pixels = image.load()
image_list = []
for x in range(width):
    image_list.append([])
    for y in range(height):
        image_list[x].append(image.getpixel((y,x)))

def gaussian_filter(size, var):
    return [[gauss(i - size//2, j - size//2,var) for i in range(size)]
                                                  for j in range(size)]

def gauss(x, y, var):
    return 1/(2*pi*var)*e**(-(x**2+y**2)/(2*var))

filter = gaussian_filter(7, 1.6)

plot_transform_2d(add_zero(image_list), use_log=False)
image_list, filter, pad_len = pad_zeros(image_list, filter)

image_transform = dft2d(center_2d_transform(image_list), 1)
filter_transform = dft2d(center_2d_transform(filter), 1)
plot_transform_2d(image_transform)
plot_transform_2d(filter_transform)

filtered_image_freq = complex_hadamard(image_transform, filter_transform)

image_inverse_transform = dft2d(filtered_image_freq, -1)
remove_centering_2d_transform(image_inverse_transform)

filtered_image = unpad(image_inverse_transform, pad_len)

plot_transform_2d(filtered_image, use_log=False)
