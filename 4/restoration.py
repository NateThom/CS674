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

def get_magnitude(data, use_log=True, square=False):
    mag = []
    for i in range(len(data)):
        mag.append([])
        for j in range(len(data[i]) // 2):
            if square:
                mag[i].append((data[i][2*j]**2) + (data[i][2*j + 1]**2))
            elif use_log:
                mag[i].append(log(((data[i][2*j]**2) + (data[i][2*j + 1]**2))**(1/2) + 1))
            else:
                mag[i].append(((data[i][2*j]**2) + (data[i][2*j + 1]**2))**(1/2))

    return mag

def plot_transform_2d(data, use_log=True):
    mag = get_magnitude(data, use_log)

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
            real_part = a[i][2*j]*b[i][2*j] - a[i][2*j+1]*b[i][2*j+1]
            complex_part = a[i][2*j]*b[i][2*j+1] + a[i][2*j+1]*b[i][2*j]
            result[-1].append(real_part)
            result[-1].append(complex_part)

    return result

def complex_division(a, b, cutoff=30):
    result = []
    for i in range(len(a)):
        result.append([])
        for j in range(len(a[0])//2):
            if ((j-len(a)//2)**2 + (i-len(a)//2)**2)**(1/2) > cutoff:
                result[-1].append(a[i][2*j])
                result[-1].append(a[i][2*j+1])
                continue
            real_part = (a[i][2*j]*b[i][2*j] + a[i][2*j+1]*b[i][2*j+1]) \
                        / (b[i][2*j]**2 + b[i][2*j+1]**2)
            complex_part = (a[i][2*j+1]*b[i][2*j] - a[i][2*j]*b[i][2*j+1]) \
                           / (b[i][2*j]**2 + b[i][2*j+1]**2)
            result[-1].append(real_part)
            result[-1].append(complex_part)

    return result

def complex_invert(matrix, cutoff=30):
    result = []
    for i in range(len(matrix)):
        result.append([])
        for j in range(len(matrix[0])//2):
            if ((j-len(matrix)//2)**2 + (i-len(matrix)//2)**2)**(1/2) > cutoff:
                result[-1].append(1)
                result[-1].append(1)
                continue
            real_part = matrix[i][2*j] / (matrix[i][2*j]**2 + matrix[i][2*j+1]**2)
            complex_part = -matrix[i][2*j+1] / (matrix[i][2*j]**2 + matrix[i][2*j+1]**2)
            result[-1].append(real_part)
            result[-1].append(complex_part)

    return result

def complex_square(matrix):
    result = []
    for i in range(len(matrix)):
        result.append([])
        for j in range(len(matrix[0])//2):
            real_part = matrix[i][2*j]**2 - matrix[i][2*j+1]**2
            complex_part = 2*matrix[i][2*j]*matrix[i][2*j+1]
            result[-1].append(real_part)
            result[-1].append(complex_part)

    return result

def add_scalar(matrix, scalar):
    result = []
    for i in range(len(matrix)):
        result.append([])
        for j in range(len(matrix[0])//2):
            real_part = matrix[i][2*j] + scalar
            complex_part = matrix[i][2*j+1]
            result[-1].append(real_part)
            result[-1].append(complex_part)

    return result

def pad_zeros(image, filter, pad_value=0):
    n = len(image)
    m = len(filter)

    # Padding must be at least n + m - 1
    min_padding = n + m -1
    # Pad to a power of 2
    padding = 0
    exponent = 1
    while padding < min_padding:
        padding = 2**exponent
        exponent += 1

    for i in range(m,padding):
        filter.append([0 for _ in range(padding)])
    for i in range(m):
        filter[i].extend([0 for _ in range(padding-m)])

    for i in range(n,padding):
        image.append([pad_value for _ in range(padding)])
    for i in range(n):
        image[i].extend([pad_value for _ in range(padding-n)])

    return image, filter, padding - n

def unpad(image, pad_len):
    if pad_len == 0:
        return image
    for i in range(pad_len):
        image[i] = image[i][:pad_len]

    return image[:pad_len]

def gaussian_filter(size, var):
    return [[gauss(i - size//2, j - size//2,var) for i in range(size)]
                                                  for j in range(size)]

def gauss(x, y, var):
    return 1/(2*pi*var)*e**(-(x**2+y**2)/(2*var))

def normalize_0_255(input_list):
    min_v = min([min(i) for i in input_list])
    max_v = max([max(i) for i in input_list])

    for w in range(len(input_list)):
        for h in range(len(input_list[w])):
            input_list[w][h] = int((input_list[w][h] - min_v)
                                   * (255 / (max_v - min_v)))

    return input_list

def list_to_image(image_list):
    height = len(image_list)
    width = len(image_list[0])
    output_image = Image.new("L", (width, height))
    output_image_pixels = output_image.load()
    for i in range(height):
        for j in range(width):
            output_image_pixels[j,i] = int(image_list[i][j])
    return output_image

def get_image_list(path):
    image = Image.open(path)

    width, height = image.size
    image_pixels = image.load()
    image_list = []
    for x in range(width):
        image_list.append([])
        for y in range(height):
            image_list[x].append(image.getpixel((y,x)))

    return image_list

def motion_blur_filter(T, a, b, size):
    p1 = lambda u,v: T/(pi*(u*a + v*b + 0.00001))
    real_p = lambda u,v: p1(u,v)*1/2*sin(2*pi*(u*a + v*b + 0.00001))
    imaginary_p = lambda u,v: -p1(u,v)*sin(pi*(u*a + v*b + 0.00001))**2

    filter = []
    for i in range(-size//2+1, size//2+1):
        filter.append([])
        for j in range(-size//2+1, size//2+1):
            real_part = real_p(i,j)
            im_part = imaginary_p(i,j)
            #filter[-1].append(real_part)
            #filter[-1].append(im_part)
            if real_part > 0:
                filter[-1].append(max(real_part, .05))
            else:
                filter[-1].append(min(real_part, -.05))
            if im_part > 0:
                filter[-1].append(max(im_part, .05))
            else:
                filter[-1].append(min(im_part, -.05))

    return filter

def normalize_2d(input_list):
    list_sum = 0
    for i in input_list:
        list_sum += sum(i)

    output_list = [[float(input_list[i][j])/list_sum
                    for j in range(len(input_list[i]))]
                   for i in range(len(input_list))]

    return output_list

def add_noise(matrix, mu, sigma):
    result = []
    for i in range(len(matrix)):
        result.append([])
        for j in range(len(matrix[0])):
            result[-1].append(max(min(int(matrix[i][j]
                                      + np.random.normal(mu,sigma)), 255),0))

    return result

edges_test = False
apply_motion_blur = True
remove_motion_blur = True
remove_motion_blur_wiener = True

if edges_test:
    filter = [[1,1,1],[0,0,0],[-1,-1,-1]]
    nate = get_image_list("nate.pgm")

    nate, filter, pad_len = pad_zeros(nate, filter)
    centered_nate = center_2d_transform(nate)
    centered_filter = center_2d_transform(filter)

    image_transform = dft2d(centered_nate, 1)
    filter_transform = dft2d(centered_filter, 1)

    filtered_image_freq = complex_hadamard(image_transform, filter_transform)

    plot_transform_2d(image_transform)
    plot_transform_2d(filter_transform)
    plot_transform_2d(filtered_image_freq)

    image_inverse_transform = dft2d(filtered_image_freq, -1)

    uncentered = remove_centering_2d_transform(image_inverse_transform)

    filtered_image = unpad(uncentered, pad_len)
    plt.imshow(filtered_image, cmap="gray")
    plt.show()

    filtered_output = list_to_image(filtered_image)
    filtered_output.save("nate-gauss.jpg")

if apply_motion_blur:
    nate = get_image_list("lenna.pgm")
    noisy_nate = add_noise(nate, 0, 0)
    centered_nate = center_2d_transform(noisy_nate)
    image_transform = dft2d(centered_nate, 1)

    filter = motion_blur_filter(1, .1, .1, 256)
    filtered_image = complex_hadamard(image_transform, filter)

    filtered_nate = dft2d(filtered_image, -1)

    new_nate = remove_centering_2d_transform(filtered_nate)

    nate_out = list_to_image(normalize_0_255(new_nate))
    nate_out.save("lenna-motion.jpg")

    noisy_nate_out = list_to_image(noisy_nate)
    noisy_nate_out.save("lenna-noisy.jpg")

if remove_motion_blur:
    nate = get_image_list("lenna-motion.jpg")
    centered_nate = center_2d_transform(nate)
    image_transform = dft2d(centered_nate, 1)

    filter = motion_blur_filter(1, .1, .1, 256)

    #plot_transform_2d(filter)
    #plot_transform_2d(complex_invert(filter, cutoff=900))

    for cutoff in [10, 40, 70, 100]:
        invert_filter = complex_invert(filter, cutoff=cutoff)

        filtered_image = complex_hadamard(image_transform, invert_filter)

        filtered_nate = dft2d(filtered_image, -1)

        new_nate = remove_centering_2d_transform(filtered_nate)

        nate_out = list_to_image(normalize_0_255(new_nate))
        nate_out.save("lenna-restore-{}.jpg".format(cutoff))

if remove_motion_blur_wiener:
    nate = get_image_list("lenna-motion.jpg")
    centered_nate = center_2d_transform(nate)
    image_transform = dft2d(centered_nate, 1)

    filter = motion_blur_filter(1, .1, .1, 256)
    filter_square = add_zero(get_magnitude(filter, square=True))

    for k in [.09, .07, .05, .03, .01, .001]:
        noise_term = complex_division(filter_square,
                                      add_scalar(filter_square, k), cutoff=500)
        inverse_filtered = complex_division(image_transform, filter, cutoff=500)
        filtered_image = complex_hadamard(noise_term, inverse_filtered)

        filtered_nate = dft2d(filtered_image, -1)

        new_nate = remove_centering_2d_transform(filtered_nate)

        nate_out = list_to_image(normalize_0_255(new_nate))
        nate_out.save("lenna-restore-wiener-{}.jpg".format(k))
