import fft
import matplotlib.pyplot as plt
from math import cos, pi

fft_func = fft.FFT()

experiment1 = True
experiment2 = True
experiment3 = True

# Experiment 1
# Part 1
def center_transform(data):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i]*(-1)**(i))
    return new_data

def plot_transform_1d(data, centered=True):
    if centered:
        x = range(-len(data)//4, len(data)//4)
    else:
        x = range(len(data)//2)
    mag = [(data[2*i]**2 + data[2*i+1]**2)**(1/2) for i in range(len(data)//2)]
    plt.plot(x, [data[2*i] for i in range(len(data)//2)], label="real")
    plt.plot(x, [data[2*i+1] for i in range(len(data)//2)], label="imaginary")
    plt.plot(x, mag, label="magnitude")
    plt.axhline(0, lw=1, color="grey")
    plt.legend()
    plt.show()

def gen_cos_wave(u, N):
    return [cos(2*pi*u*x/N) for x in range(N)]

if experiment1:
    data = [2, 3, 4, 4]
    transform = fft_func.fourier_transform(data, 1)
    plot_transform_1d(transform, centered=False)
    print(transform)

    inverse_transform = fft_func.fourier_transform(transform, -1)
    print(inverse_transform)

    # Part 2
    cos_wave = gen_cos_wave(8, 128)
    plt.plot(cos_wave)
    plt.show()
    cos_transform = fft_func.fourier_transform(center_transform(cos_wave), 1)
    plot_transform_1d(cos_transform)

    # Part 3
    rect_func = []
    with open("Rect_128.dat") as rect_file:
        for line in rect_file:
            rect_func.append(float(line))
    plt.plot(rect_func)
    plt.show()

    rect_transform = fft_func.fourier_transform(center_transform(rect_func), 1)
    plot_transform_1d(rect_transform)

# Experiment 2

# Experiment 3
