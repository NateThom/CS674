import fft
import math

fft_func = fft.FFT()

# Be sure to leave a zero in every other space in the array to account for imaginary numbers
# data = [1, 0, 2, 0, 3, 0, 4, 0]

# Pass 1 for forward, -1 for inverse
# transform = fft_func.fourier_transform(data, 1)
# inv_transform = fft_func.fourier_transform(transform, -1)

# print(data)
# print(transform)
# print(inv_transform)


# Experiment 1
# Part 1
data = [2, 0, 3, 0, 4, 0, 4, 0]
transform = fft_func.fourier_transform(data, 1)

magnitude = []
phase = []
for i in range(0, len(transform), 2):
    magnitude.append(math.sqrt(transform[i]**2 + transform[i+1]**2))
    phase.append(math.atan2(transform[i+1]/transform[i]))

# re = mag .* cos(phase);
# im = mag .* sin(phase);

inverse_transform = fft_func.fourier_transform(transform, -1)

# Experiment 2

# Experiment 3
