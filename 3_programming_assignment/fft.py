import subprocess
import os

class FFT():
    def __init__(self):
        if not os.path.isfile("fft"):
            subprocess.call(["g++", "fft.c", "-o", "fft"])

    def fourier_transform(self, data, mode, extend=True):
        # If not otherwise specified, assume the input to
        # forward transform does not have zeros where the
        # imaginary values will be
        if extend == True and mode == 1:
            # add a zero after every real value in list
            data_complex = [0]*len(data)*2
            data_complex[::2] = data
        else:
            data_complex = data

        output = subprocess.check_output(["./fft"] + [str(d) for d in data_complex]
                                         + [str(mode)]).split()
        transform = [float(t) for t in output]

        # Get rid of imaginary values for inverse transform
        if mode == -1:
            for i in range(1, len(transform), 2):
                transform[i] = 0

        return transform
