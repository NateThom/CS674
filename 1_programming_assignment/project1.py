from PIL import Image
import os
import glob
import matplotlib.pyplot as plt

def sample(image, factor):
    image_width, image_height = image.size

    new_pixels = []
    for height_index in range(image_height // factor):
        for width_index in range(image_width // factor):
            new_pixels.append(image.getpixel((width_index * factor, height_index * factor)))

    new_image = Image.new('L', (image_width // factor, image_height // factor))
    new_image.putdata(new_pixels)
    new_image = new_image.resize((image_width, image_height))

    return new_image

def change_quantization(image, levels):
    image_width, image_height = image.size

    divisor = 256 // levels
    for i in range(image_width):
        for j in range(image_height):
            image.putpixel((i,j), (image.getpixel((i,j))//divisor) * divisor)

    return image

def get_histogram(image):
    image_width, image_height = image.size
    histogram = [0] * 256
    for i in range(image_width):
        for j in range(image_height):
            histogram[image.getpixel((i,j))] += 1

    return histogram

def plot_histogram(histogram, name):
    plt.plot(histogram)
    plt.title(name)
    plt.ylim(0)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.savefig("histograms/" + name + ".png")
    plt.close()

def histogram_eq(image, map):
    image_width, image_height = image.size
    for i in range(image_width):
        for j in range(image_height):
            image.putpixel((i,j), map[image.getpixel((i,j))])
            # print(image.getpixel((i,j)))
            # print(map[image.getpixel((i,j))])
            # print()

def histogram_norm(image, name):
    image_width, image_height = image.size

    histogram = get_histogram(image)
    plot_histogram(histogram, name + "-eq-in")

    pixel_map = [0] * 256
    corrected_pixels = 0

    sum_r = image_width * image_height
    for i in range(256):
        corrected_pixels += histogram[i]
        pixel_map[i] = (corrected_pixels*255) // sum_r

    histogram_eq(image, pixel_map)

    plot_histogram(get_histogram(image), name + "-eq-out")

    return image

def histogram_spec(image, path_to_image_histogram, name):
    image_width, image_height = image.size

    histogram = get_histogram(image)
    plot_histogram(histogram, name + "-spec-in")

    target_dist_image = Image.open(path_to_image_histogram)
    target_dist_hist = get_histogram(target_dist_image)
    plot_histogram(target_dist_hist, "spec-dist-" + path_to_image_histogram.split("/")[-1])

    pixel_map_1 = [0] * 256
    corrected_pixels_1 = 0

    sum_r = image_width * image_height
    for i in range(256):
        corrected_pixels_1 += histogram[i]
        pixel_map_1[i] = (corrected_pixels_1*255) // sum_r

    pixel_map_2 = [0] * 256
    corrected_pixels_2 = 0

    for i in range(256):
        corrected_pixels_2 += target_dist_hist[i]
        pixel_map_2[i] = (corrected_pixels_2 * 255) // sum_r

    inverse_mapping_dict = {}
    for i in range(256):
        inverse_mapping_dict[pixel_map_2[i]] = i

    pixel_map_3 = [0] * 256
    for i in range(256):
        if pixel_map_1[i] in inverse_mapping_dict:
            pixel_map_3[i] = inverse_mapping_dict[pixel_map_1[i]]
        else:
            closest = min(list(inverse_mapping_dict.keys()), key=lambda x: abs(x-pixel_map_1[i]))
            pixel_map_3[i] = inverse_mapping_dict[closest]

    histogram_eq(image, pixel_map_3)

    plot_histogram(get_histogram(image), name + "-spec-out-dist-" + path_to_image_histogram.split("/")[-1])

    return image

for part in range(4):
    if not os.path.exists("part{}_output".format(part+1)):
        os.makedirs("part{}_output".format(part+1))
if not os.path.exists("histograms"):
    os.makedirs("histograms")
if not os.path.exists("part2_eq"):
    os.makedirs("part2_eq")

for image_path in glob.glob("../images-pgm/*"):
    basename = image_path.split("/")[-1].split(".")[0]
    image = Image.open(image_path)

    normalized = histogram_norm(image.copy(), basename)

    sample(image,2).save("part1_output/{}-{}.pgm".format(basename,2))
    sample(image,4).save("part1_output/{}-{}.pgm".format(basename,4))
    sample(image,8).save("part1_output/{}-{}.pgm".format(basename,8))

    change_quantization(image.copy(),128).save("part2_output/{}-{}.pgm".format(basename,128))
    change_quantization(image.copy(),32).save("part2_output/{}-{}.pgm".format(basename,32))
    change_quantization(image.copy(),8).save("part2_output/{}-{}.pgm".format(basename,8))
    change_quantization(image.copy(),2).save("part2_output/{}-{}.pgm".format(basename,2))

    change_quantization(normalized.copy(),128).save("part2_eq/{}-{}.pgm".format(basename,128))
    change_quantization(normalized.copy(),32).save("part2_eq/{}-{}.pgm".format(basename,32))
    change_quantization(normalized.copy(),8).save("part2_eq/{}-{}.pgm".format(basename,8))
    change_quantization(normalized.copy(),2).save("part2_eq/{}-{}.pgm".format(basename,2))

    histogram_norm(image, basename).save("part3_output/{}.pgm".format(basename))

    image = Image.open(image_path)
    histogram_spec(image, "../images-pgm/sf.pgm", basename).save("part4_output/{}.pgm".format(basename + "_sf_dist"))
    image = Image.open(image_path)
    histogram_spec(image, "../images-pgm/peppers.pgm", basename).save("part4_output/{}.pgm".format(basename + "_peppers_dist"))
