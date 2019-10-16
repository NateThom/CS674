from PIL import Image

def normalize(input_list):
    list_sum = 0
    for i in input_list:
        list_sum += sum(i)

    output_list = [[float(input_list[i][j])/list_sum for j in range(len(input_list[i]))] for i in range(len(input_list))]

    return output_list

def image_to_list(image):
    width, height = image.size
    output_list = []
    for i in range(width):
        output_list.append([])
        for j in range(height):
            output_list.append(image.getpixel((i,j)))
    return output_list

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
                        weighted_sum += pixels[i+h_offset, j+v_offset]*convolution[h_offset][v_offset]
            pixels_new[i,j] = int(weighted_sum)

    return new_img

def correlate(image, correlation, padding=0):
    pixels = image.load()
    width, height = image.size
    conv_size = len(correlation)

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
                        weighted_sum += pixels[i+h_offset, j+v_offset]*correlation[n][k]
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
    filter = image_to_list(filter)

image = Image.open("../images-pgm/Image.pgm")

# filter = [[1/9]*3]*3

# convolve(image, filter).save("convolution.png")
# correlate(image, filter).save("correlation.png")


# image = Image.open("../images-pgm/lenna.pgm")
#
# gaus_mask_7_7 = [[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]]
#
# gaus_mask_8_8 = [[2,2,3,4,5,5,6,6,6,5,5,4,3,2,2],[2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],[3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],
#                  [4,5,7,9,11,12,13,13,13,12,11,9,7,5,4],[5,7,9,10,13,14,15,16,15,14,13,10,9,7,5],[5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],
#                  [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],[6,8,11,13,16,18,19,20,19,18,16,13,11,8,6],[6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],
#                  [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],[5,7,9,10,13,14,15,16,15,14,13,10,9,7,5],[4,5,7,9,11,12,13,13,13,12,11,9,7,5,4],
#                  [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],[2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],[2,2,3,4,5,5,6,6,6,5,5,4,3,2,2]]
#
# norm_gaus_mask_7_7 = normalize(gaus_mask_7_7)
# norm_gaus_mask_8_8 = normalize(gaus_mask_8_8)
#
# correlate(image, norm_gaus_mask_7_7).save("smoothing_7_7.png")
# correlate(image, norm_gaus_mask_8_8).save("smoothing_8_8.png")

median_input = int(input("Enter the size of your median filter: "))
