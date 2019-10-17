from PIL import Image
from math import pi, e
from random import randrange

def normalize_1d(input_list):
    output_list = [float(input_list[i])/sum(input_list) for i in range(len(input_list))]

    return output_list

def normalize_2d(input_list):
    list_sum = 0
    for i in input_list:
        list_sum += sum(i)

    output_list = [[float(input_list[i][j])/list_sum for j in range(len(input_list[i]))] for i in range(len(input_list))]

    return output_list

def normalize_0_255(input_list):
    min = 999999
    max = 0
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

def image_to_list(image):
    width, height = image.size
    output_list = []
    for i in range(width):
        output_list.append([])
        for j in range(height):
            output_list[i].append(image.getpixel((i,j)))
    return output_list

def list_to_image(image_list, image):
    width = len(image_list)
    height = len(image_list[0])
    output_image = Image.new(image.mode, image.size)
    output_image_pixels = output_image.load()
    for i in range(width):
        for j in range(height):
            output_image_pixels[i, j] = image_list[i][j]
    return output_image

def gaussian_filter(size, var):
    return [[gauss(i - size//2, j - size//2,var) for i in range(size)]
                                                  for j in range(size)]

def gauss(x, y, var):
    return 1/(2*pi*var)*e**(-(x**2+y**2)/(2*var))

def salt_pepper_noise(image, percent=30):
    width, height = image.size

    new_image = image.copy()
    new_image_pixels = new_image.load()

    white_pixel_max = percent // 2

    for i in range(width):
        for j in range(height):
            random_number = randrange(100)
            if 0 <= random_number <= white_pixel_max:
                new_image_pixels[i, j] = 255
            elif white_pixel_max < random_number <= percent:
                new_image_pixels[i, j] = 0

    return new_image

def convolve(image, convolution, padding=0):
    pixels = image.load()
    width, height = image.size
    conv_size = len(convolution)

    new_img = Image.new(image.mode, image.size)
    pixels_new = new_img.load()

    for i in range(width):
        print(i)
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
                        weighted_sum += pixels[i+h_offset, j+v_offset]*convolution[n][k]
            pixels_new[i,j] = int(weighted_sum)

    return new_img

def correlate(image, correlation, padding=0, median_flag=0):
    image_list = image_to_list(image)

    # pixels = image.load()
    width, height = image.size
    corr_width = len(correlation)
    corr_height = len(correlation[0])
    pixels = image_list
    width = len(image_list)
    height = len(image_list[0])

    new_image_list = []

    # new_image = Image.new(image.mode, image.size)
    # pixels_new = new_image.load()

    for i in range(width):
        print(i)
        new_image_list.append([])
        for j in range(height):

            if median_flag is 1:
                neighborhood_pixels = []

            weighted_sum = 0
            for n in range(corr_width):
                for k in range(corr_height):
                    v_offset = n - corr_width // 2
                    h_offset = k - corr_height // 2
                    if i + h_offset < 0 or i + h_offset >= width\
                                        or j + v_offset < 0\
                                        or j + v_offset >= height:
                        if median_flag is 1:
                            neighborhood_pixels.append(padding)
                        else:
                            weighted_sum += padding
                    else:
                        # weighted_sum += pixels[i + h_offset, j + v_offset] * correlation[n][k]
                        if median_flag is 1:
                            neighborhood_pixels.append(pixels[i + h_offset][j + v_offset])
                        else:
                            weighted_sum += pixels[i+h_offset][j+v_offset]*correlation[n][k]
            if median_flag is 1:
                new_image_list[i].append(sorted(neighborhood_pixels)[len(neighborhood_pixels)//2])
            else:
                new_image_list[i].append(int(weighted_sum))
            # pixels_new[i,j] = int(weighted_sum)
    if median_flag is 0:
        new_image_list = normalize_0_255(new_image_list)
    new_image = list_to_image(new_image_list, image)
    return new_image


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
    filter = image_to_list(filter)
    # filter = normalize_2d(filter)

# image = Image.open("../images-pgm/Image.pgm")

# correlate(image, filter).save("correlation_norm_filter.png")

image = Image.open("../images-pgm/lenna.pgm")

# gaus_mask_7_7 = [[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]]

# gaus_mask_8_8 = [[2,2,3,4,5,5,6,6,6,5,5,4,3,2,2],[2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],[3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],
#                  [4,5,7,9,11,12,13,13,13,12,11,9,7,5,4],[5,7,9,10,13,14,15,16,15,14,13,10,9,7,5],[5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],
#                  [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],[6,8,11,13,16,18,19,20,19,18,16,13,11,8,6],[6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],
#                  [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],[5,7,9,10,13,14,15,16,15,14,13,10,9,7,5],[4,5,7,9,11,12,13,13,13,12,11,9,7,5,4],
#                  [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],[2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],[2,2,3,4,5,5,6,6,6,5,5,4,3,2,2]]

# norm_gaus_mask_7_7 = normalize_2d(gaus_mask_7_7)
# norm_gaus_mask_8_8 = normalize_2d(gaus_mask_8_8)

# correlate(image, norm_gaus_mask_7_7).save("smoothing_7_7.png")
# correlate(image, norm_gaus_mask_8_8).save("smoothing_8_8.png")

image = salt_pepper_noise(image)
image.save("salt_pepper.png")

median_input = int(input("Enter the size of your median filter: "))
filter = [[1]*median_input]*median_input
correlate(image, filter, median_flag=1).save("median_filtering.png")
