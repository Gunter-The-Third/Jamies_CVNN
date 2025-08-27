import numpy as np
from cvnn import Sequential
from cvnn.layers import ComplexDense
from cvnn.activations import complex_relu, complex_tanh, jam, jam_derivative, complex_relu_backward, complex_tanh_backward
from cvnn.initialisations import jamie


# Complex XOR dataset: x = [1+1j], [1-1j], [-1+1j], [-1-1j] (single input dim)
Xc = np.array([
    [ 1+1j],
    [ 1-1j],
    [-1+1j],
    [-1-1j],
], dtype=np.complex128)
# XOR on sign of real and imag parts
Yc = np.array([
    [0],  # (1, 1) -> 0
    [1],  # (1, -1) -> 1
    [1],  # (-1, 1) -> 1
    [0],  # (-1, -1) -> 0
], dtype=np.complex128)

# Multilayer model for XOR
model = Sequential([
    ComplexDense(input_dim=1, output_dim=8, weight_init=jamie),
    (complex_tanh, complex_tanh_backward),
    ComplexDense(input_dim=8, output_dim=1, weight_init=jamie),
    (jam, jam_derivative)
])

print("Training on XOR dataset...")
history = model.fit(Xc, Yc, epochs=1000, lr=0.1, return_history=True, track=['predictions'])

# Evaluate
out = model.predict(Xc)
print("Predictions (real part):", out.real.round(2))
print("Targets:", Yc.T)


# Show loss history summary
print("First 10 losses:", history['loss'][:10])
print("Last 10 losses:", history['loss'][-10:])

# Plot loss if requested and matplotlib is available
import sys
if '--plot' in sys.argv:
    try:
        import matplotlib.pyplot as plt
        plt.plot(history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.show()
    except ImportError:
        print('matplotlib not installed, cannot plot.')
