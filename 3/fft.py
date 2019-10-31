import subprocess
import os

class FFT():
    def __init__(self):
        if not os.path.isfile("fft"):
            subprocess.call(["g++", "fft.c", "-o", "fft"])

    def fourier_transform(self, data, mode):
        output = subprocess.check_output(["./fft"] + [str(d) for d in data]
                                         + [str(mode)]).split()
        transform = [float(t) for t in output]

        # Get rid of imaginary values for inverse transform
        if mode == -1:
            for i in range(1, len(transform), 2):
                transform[i] = 0
        return transform
