import fft

fft_func = fft.FFT()

data = [1, 0, 2, 0, 3, 0, 4, 0]
transform = fft_func.fourier_transform(data, 1)
inv_transform = fft_func.fourier_transform(transform, -1)

print(data)
print(transform)
print(inv_transform)
