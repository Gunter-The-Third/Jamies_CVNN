import numpy as np

class ComplexDense:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim) + 1j * np.random.randn(input_dim, output_dim)
        self.b = np.zeros((1, output_dim), dtype=np.complex128)

    def forward(self, x):
        return x @ self.W + self.b
