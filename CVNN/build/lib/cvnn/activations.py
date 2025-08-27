import numpy as np

def complex_relu(z):
    return np.maximum(z.real, 0) + 1j * np.maximum(z.imag, 0)
