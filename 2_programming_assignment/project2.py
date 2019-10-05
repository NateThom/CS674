from PIL import Image

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
                        weighted_sum += pixels[i+h_offset, j+v_offset]*convolution[h_offset][v_offset]
            pixels_new[i,j] = int(weighted_sum)

    return new_img

image = Image.open("nate.pgm")

filter = [[1/9]*3]*3

convolve(image, filter).save("nate-improved.png")
