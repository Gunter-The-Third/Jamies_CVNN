import numpy as np
from cvnn import Sequential
from cvnn.layers import ComplexDense
from cvnn.activations import complex_relu, complex_tanh

# Dummy data: learn a random linear mapping
np.random.seed(42)
x = np.random.randn(100, 4) + 1j * np.random.randn(100, 4)
W_true = np.random.randn(4, 2) + 1j * np.random.randn(4, 2)
y = x @ W_true

# Multilayer model
def make_model():
    return Sequential([
        ComplexDense(input_dim=4, output_dim=8),
        complex_relu,
        ComplexDense(input_dim=8, output_dim=2),
        complex_tanh
    ])

model = make_model()

print("Training multilayer network...")
model.fit(x, y, epochs=50, lr=0.01)

# Evaluate
out = model.predict(x)
mse = np.mean(np.abs(out - y) ** 2)
print(f"Final MSE: {mse}")
