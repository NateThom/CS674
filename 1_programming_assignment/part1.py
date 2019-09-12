from PIL import Image


def open(filepath):
    # Read in image and convert it to list
    lenna = Image.open(filepath)

    # lenna_data = lenna_img.getdata()
    # lenna_list = []

    # for img_col in range(0, 256):
    #     lenna_list.append([])
    #     for img_row in range(0, 256):
    #         lenna_list[img_col].append(lenna_data[(img_col*img_col)+img_row])

    pixels = list(lenna.getdata())
    width, height = lenna.size
    pixels = [pixels[height_index * width:(height_index + 1) * width] for height_index in range(height)]

    lenna.close()

    return pixels, width, height


def save_image(image, filepath, extension):
    image.save(filepath, extension)


def subsample(input_filepath, output_filepath, output_extension, factor):
    image, image_width, image_height = open(input_filepath)

    new_pixels = []
    for width_index in range(int(image_width / factor)):
        # new_pixels.append([])
        # for height_index in range(int(height/factor)):
        #     new_pixels[width_index].append(pixels[width_index*factor][height_index*factor])
        for height_index in range(int(image_height / factor)):
            new_pixels.append(image[width_index * factor][height_index * factor])

    new_image = Image.new('L', (int(image_width / factor), int(image_height / factor)))
    new_image.putdata(new_pixels)
    new_image = new_image.resize((image_width, image_height))

    save_image(new_image, output_filepath, output_extension)

    return new_image

subsample("../images/lenna.gif", "./part1_output/lenna_2", "gif", 2)
subsample("../images/lenna.gif", "./part1_output/lenna_4", "gif", 4)
subsample("../images/lenna.gif", "./part1_output/lenna_8", "gif", 8)

subsample("../images/aerial.gif", "./part1_output/aerial_2", "gif", 2)
subsample("../images/aerial.gif", "./part1_output/aerial_4", "gif", 4)
subsample("../images/aerial.gif", "./part1_output/aerial_8", "gif", 8)

subsample("../images/boat.gif", "./part1_output/boat_2", "gif", 2)
subsample("../images/boat.gif", "./part1_output/boat_4", "gif", 4)
subsample("../images/boat.gif", "./part1_output/boat_8", "gif", 8)

subsample("../images/f_16.gif", "./part1_output/f_16_2", "gif", 2)
subsample("../images/f_16.gif", "./part1_output/f_16_4", "gif", 4)
subsample("../images/f_16.gif", "./part1_output/f_16_8", "gif", 8)

subsample("../images/lax.gif", "./part1_output/lax_2", "gif", 2)
subsample("../images/lax.gif", "./part1_output/lax_4", "gif", 4)
subsample("../images/lax.gif", "./part1_output/lax_8", "gif", 8)

subsample("../images/peppers.gif", "./part1_output/peppers_2", "gif", 2)
subsample("../images/peppers.gif", "./part1_output/peppers_4", "gif", 4)
subsample("../images/peppers.gif", "./part1_output/peppers_8", "gif", 8)

subsample("../images/sf.gif", "./part1_output/sf_2", "gif", 2)
subsample("../images/sf.gif", "./part1_output/sf_4", "gif", 4)
subsample("../images/sf.gif", "./part1_output/sf_8", "gif", 8)

subsample("../images/tools.gif", "./part1_output/tools_2", "gif", 2)
subsample("../images/tools.gif", "./part1_output/tools_4", "gif", 4)
subsample("../images/tools.gif", "./part1_output/tools_8", "gif", 8)

subsample("../images/wheel.gif", "./part1_output/wheel_2", "gif", 2)
subsample("../images/wheel.gif", "./part1_output/wheel_4", "gif", 4)
subsample("../images/wheel.gif", "./part1_output/wheel_8", "gif", 8)

# aerial, aerial_width, aerial_height = open("../images/aerial.gif")
# aerial_factor_2 = subsample(2, aerial, aerial_width, aerial_height)
# save_image(aerial_factor_2, "part1_output", "aerial_2", "gif")
# aerial_factor_4 = subsample(4, aerial, aerial_width, aerial_height)
# save_image(aerial_factor_4, "part1_output", "aerial_4", "gif")
# aerial_factor_8 = subsample(8, aerial, aerial_width, aerial_height)
# save_image(aerial_factor_8, "part1_output", "aerial_8", "gif")
