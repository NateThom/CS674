from PIL import Image
import os
import glob

def subsample(input_filepath, output_filepath, output_extension, factor):
    image = Image.open(input_filepath)
    pixels = image.load()
    image_width, image_height = image.size

    new_pixels = []
    for height_index in range(image_height // factor):
        for width_index in range(image_width // factor):
            new_pixels.append(pixels[width_index * factor, height_index * factor])

    new_image = Image.new('L', (image_width // factor, image_height // factor))
    new_image.putdata(new_pixels)
    new_image = new_image.resize((image_width, image_height))

    new_image.save(output_filepath, output_extension)

    return new_image

def change_quantization(input_filepath, output_filepath, levels):
    image = Image.open(input_filepath)
    pixels = image.load()
    image_width, image_height = image.size

    divisor = 256 // levels
    for i in range(image_width):
        for j in range(image_height):
            pixels[i,j] = (pixels[i,j]//divisor) * divisor

    image.save(output_filepath)

if not os.path.exists("part1_output"):
    os.makedirs("part1_output")

if not os.path.exists("part2_output"):
    os.makedirs("part2_output")

for image_path in glob.glob("../images/*"):
    basename = image_path.split("/")[-1].split(".")[0]

    subsample(image_path, "part1_output/{}-{}.gif".format(basename,2), "gif", 2)
    subsample(image_path, "part1_output/{}-{}.gif".format(basename,4), "gif", 4)
    subsample(image_path, "part1_output/{}-{}.gif".format(basename,8), "gif", 8)

    change_quantization(image_path, "part2_output/{}-{}.gif".format(basename,128), 128)
    change_quantization(image_path, "part2_output/{}-{}.gif".format(basename,32), 32)
    change_quantization(image_path, "part2_output/{}-{}.gif".format(basename,8), 8)
    change_quantization(image_path, "part2_output/{}-{}.gif".format(basename,2), 2)
