from PIL import Image

import numpy as np

def convolve(image, convolution, padding=0):
    pixels = image.load()
    width, height = image.size
    conv_size = len(convolution)

    new_img = Image.new(image.mode, image.size)
    pixels_new = new_img.load()

    for i in range(width):
        for j in range(height):
            weighted_sum = 0
            for n in range(conv_size):
                for k in range(conv_size):
                    v_offset = n - conv_size // 2
                    h_offset = k - conv_size // 2
                    if i + h_offset < 0 or i + h_offset >= width\
                                        or j + v_offset < 0\
                                        or j + v_offset >= height:
                        weighted_sum += padding
                    else:
                        weighted_sum += pixels[i+h_offset, j+v_offset]*convolution[k][n]
            pixels_new[i,j] = int(weighted_sum)

    return new_img

def correlate(image, correlation, padding=0):
    pixels = image.load()
    width, height = image.size
    conv_size = len(correlation)

    new_img = Image.new(image.mode, image.size)
    pixels_new = new_img.load()

    for i in range(width):
        for j in range(height):
            weighted_sum = 0
            for n in range(conv_size):
                for k in range(conv_size):
                    v_offset = n - conv_size // 2
                    h_offset = k - conv_size // 2
                    if i + h_offset < 0 or i + h_offset >= width\
                                        or j + v_offset < 0\
                                        or j + v_offset >= height:
                        weighted_sum += padding
                    else:
                        weighted_sum += pixels[i+h_offset, j+v_offset]*correlation[h_offset][v_offset]
            pixels_new[i,j] = int(weighted_sum)

    return new_img


correlation_input = input("Do you want to build your own correlation mask? Enter 'y' for yes, any other key for no: ")

if correlation_input is 'y':
    correlation_mask_size = int(input("Input the size of your correlation mask: "))
    filter = []
    for mask_width in range(correlation_mask_size):
        filter.append([])
        for mask_height in range(correlation_mask_size):
            filter[mask_width].append(int(input(f"Enter mask value for mask[{mask_width}][{mask_height}]: ")))
else:
    filter = Image.open("../images-pgm/Pattern.pgm")
    filter = np.array(filter)

image = Image.open("../images-pgm/Image.pgm")

# filter = [[1/9]*3]*3

convolve(image, filter).save("convolution.png")
correlate(image, filter).save("correlation.png")

