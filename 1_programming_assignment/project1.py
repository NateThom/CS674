from PIL import Image
import os
import glob

def subsample(input_filepath, output_filepath, factor):
    image = Image.open(input_filepath)
    image_width, image_height = image.size

    new_pixels = []
    for height_index in range(image_height // factor):
        for width_index in range(image_width // factor):
            new_pixels.append(image.getpixel((width_index * factor, height_index * factor)))

    new_image = Image.new('L', (image_width // factor, image_height // factor))
    new_image.putdata(new_pixels)
    new_image = new_image.resize((image_width, image_height))

    new_image.save(output_filepath)

    return new_image

def change_quantization(input_filepath, output_filepath, levels):
    image = Image.open(input_filepath)
    image_width, image_height = image.size

    divisor = 256 // levels
    for i in range(image_width):
        for j in range(image_height):
            image.putpixel((i,j), (image.getpixel((i,j))//divisor) * divisor)

    image.save(output_filepath)

def get_histogram(image, image_width, image_height):
    histogram = [0] * 256
    for i in range(image_width):
        for j in range(image_height):
            histogram[image.getpixel((i,j))] += 1

    return histogram

def histogram_eq(image, image_width, image_height, map):
    for i in range(image_width):
        for j in range(image_height):
            image.putpixel((i,j), map[image.getpixel((i,j))])

def histogram_norm(input_filepath, output_filepath):
    image = Image.open(image_path)
    image_width, image_height = image.size

    histogram = get_histogram(image, *image.size)
    pixel_map = [0] * 256
    corrected_pixels = 0

    sum_r = image_width * image_height
    for i in range(256):
        corrected_pixels += histogram[i]
        pixel_map[i] = (corrected_pixels*255) // sum_r

    histogram_eq(image, *image.size, pixel_map)
    image.save(output_filepath)

for part in range(4):
    if not os.path.exists("part{}_output".format(part+1)):
        os.makedirs("part{}_output".format(part+1))

for image_path in glob.glob("../images-pgm/*"):
    basename = image_path.split("/")[-1].split(".")[0]

    subsample(image_path, "part1_output/{}-{}.pgm".format(basename,2), 2)
    subsample(image_path, "part1_output/{}-{}.pgm".format(basename,4), 4)
    subsample(image_path, "part1_output/{}-{}.pgm".format(basename,8), 8)

    change_quantization(image_path, "part2_output/{}-{}.pgm".format(basename,128), 128)
    change_quantization(image_path, "part2_output/{}-{}.pgm".format(basename,32), 32)
    change_quantization(image_path, "part2_output/{}-{}.pgm".format(basename,8), 8)
    change_quantization(image_path, "part2_output/{}-{}.pgm".format(basename,2), 2)

    histogram_norm(image_path, "part3_output/{}.pgm".format(basename))
