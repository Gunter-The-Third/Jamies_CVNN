import numpy as np
from cvnn import Sequential
from cvnn.layers import ComplexDense
from cvnn.activations import complex_relu, complex_tanh, complex_sigmoid, jam, jam_derivative, complex_relu_backward, complex_tanh_backward, complex_sigmoid_backward
from cvnn.initialisations import jamie, jamie_bias


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

# Single layer model for XOR using JAM activation
model = Sequential([
    ComplexDense(input_dim=1, output_dim=1, weight_init=jamie),
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
        
# Multiple Tests
if '--multiple' in sys.argv:
    num_runs = 1000
    converged_count = 0
    epochs_to_converge = []

    for seed in range(num_runs):
        np.random.seed(seed)
        model = Sequential([
            ComplexDense(input_dim=1, output_dim=1, weight_init=jamie,bias_init=jamie_bias),
            (jam, jam_derivative)
        ])
        history = model.fit(Xc, Yc, epochs=1000, lr=0.1, return_history=True)
        losses = history['loss']
        # Convergence: loss < 0.01
        converged = np.any(np.array(losses) < 0.01)
        if converged:
            converged_count += 1
            first_epoch = np.where(np.array(losses) < 0.01)[0][0] + 1
            epochs_to_converge.append(first_epoch)

    print(f"Out of {num_runs} runs, {converged_count} converged.")
    if epochs_to_converge:
        print(f"Epochs to converge (min/mean/max): {min(epochs_to_converge)}/{np.mean(epochs_to_converge):.1f}/{max(epochs_to_converge)}")
    else:
        print("No runs converged.")
