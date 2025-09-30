def test_all_inits_and_activations():
    import numpy as np
    from cvnn.layers import Dense
    from cvnn.activations import (
        relu, relu_backward, complex_relu, complex_relu_backward,
        sigmoid, sigmoid_backward, complex_sigmoid, complex_sigmoid_backward,
        tanh, tanh_backward, complex_tanh, complex_tanh_backward
    )
    from cvnn.initialisations import (
        zeros, ones, normal, glorot_uniform, he_normal,
        complex_zeros, complex_ones, complex_normal, complex_glorot_uniform, complex_he_normal
    )

    # Real-valued Dense with real inits
    d_real = Dense(4, 2, weight_init=zeros, bias_init=ones, real=True)
    x_real = np.random.randn(3, 4)
    out_real = d_real.forward(x_real)
    assert out_real.shape == (3, 2)
    d_real = Dense(4, 2, weight_init=normal, bias_init=glorot_uniform, real=True)
    out_real = d_real.forward(x_real)
    d_real = Dense(4, 2, weight_init=he_normal, bias_init=zeros, real=True)
    out_real = d_real.forward(x_real)

    # Complex-valued Dense with complex inits
    d_c = Dense(4, 2, weight_init=complex_zeros, bias_init=complex_ones)
    x_c = np.random.randn(3, 4) + 1j * np.random.randn(3, 4)
    out_c = d_c.forward(x_c)
    assert out_c.shape == (3, 2)
    d_c = Dense(4, 2, weight_init=complex_normal, bias_init=complex_glorot_uniform)
    out_c = d_c.forward(x_c)
    d_c = Dense(4, 2, weight_init=complex_he_normal, bias_init=complex_zeros)
    out_c = d_c.forward(x_c)

    # Test activations (real)
    z = np.random.randn(5, 2)
    assert relu(z).shape == z.shape
    assert sigmoid(z).shape == z.shape
    assert tanh(z).shape == z.shape
    assert relu_backward(z, z).shape == z.shape
    assert sigmoid_backward(z, z).shape == z.shape
    assert tanh_backward(z, z).shape == z.shape

    # Test activations (complex)
    zc = np.random.randn(5, 2) + 1j * np.random.randn(5, 2)
    assert complex_relu(zc).shape == zc.shape
    assert complex_sigmoid(zc).shape == zc.shape
    assert complex_tanh(zc).shape == zc.shape
    assert complex_relu_backward(zc, zc).shape == zc.shape
    assert complex_sigmoid_backward(zc, zc).shape == zc.shape
    assert complex_tanh_backward(zc, zc).shape == zc.shape
    print("All real and complex initialisations and activations passed.")
import numpy as np
from cvnn.layers import Dense
from cvnn.activations import complex_relu, complex_sigmoid, complex_tanh, modrelu
from cvnn.initialisations import complex_glorot_uniform, complex_he_normal, complex_zeros, complex_ones, complex_normal, jamie
from cvnn.model import Sequential  # Force local import for correct class
def test_jamie_init():
    # Test jamie initialisation for phase constraint
    w = jamie((1000,))
    phases = np.angle(w)
    # Should be close to pi/4 or 5pi/4
    close_to_pi4 = np.abs(np.angle(np.exp(1j*(phases - np.pi/4)))) < 0.1
    close_to_5pi4 = np.abs(np.angle(np.exp(1j*(phases - 5*np.pi/4)))) < 0.1
    assert np.all(close_to_pi4 | close_to_5pi4), "Not all phases are close to pi/4 or 5pi/4"
    print("jamie initialisation passed phase constraint test.")

def test_dense():
    # Complex-valued tests
    layer = Dense(input_dim=3, output_dim=2)
    x = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
    out = layer.forward(x)
    print("Output of Dense layer (complex, default init):")
    print(out)
    assert out.shape == (5, 2), f"Expected output shape (5, 2), got {out.shape}"

    # Real-valued tests
    layer_real = Dense(input_dim=3, output_dim=2, real=True, complex=False)
    x_real = np.random.randn(5, 3)
    out_real = layer_real.forward(x_real)
    print("Output of Dense layer (real, default init):")
    print(out_real)
    assert out_real.shape == (5, 2)

def test_complex_relu():
    z = np.array([[1+2j, -1-2j], [0+0j, -3+4j]])
    out = complex_relu(z)
    print("Output of complex_relu:")
    print(out)
    # Check that negative real and imag parts are zeroed
    assert np.all(out.real >= 0) and np.all(out.imag >= 0), "complex_relu failed to zero negatives"

def test_complex_sigmoid():
    z = np.array([[1+2j, -1-2j], [0+0j, -3+4j]])
    out_sep = complex_sigmoid(z)
    print("Output of complex_sigmoid (separable):")
    print(out_sep)
    assert out_sep.shape == z.shape
    out_fc = complex_sigmoid(z, fully_complex=True)
    print("Output of complex_sigmoid (fully complex):")
    print(out_fc)
    assert out_fc.shape == z.shape

def test_complex_tanh():
    z = np.array([[1+2j, -1-2j], [0+0j, -3+4j]])
    out_sep = complex_tanh(z)
    print("Output of complex_tanh (separable):")
    print(out_sep)
    assert out_sep.shape == z.shape
    out_fc = complex_tanh(z, fully_complex=True)
    print("Output of complex_tanh (fully complex):")
    print(out_fc)
    assert out_fc.shape == z.shape

def test_modrelu():
    z = np.array([[1+2j, -1-2j], [0+0j, -3+4j]])
    out = modrelu(z, bias=0.5)
    print("Output of modrelu:")
    print(out)
    assert out.shape == z.shape

def test_real_xor():
    import numpy as np
    from cvnn.model import Sequential
    from cvnn.layers import Dense
    from cvnn.activations import complex_tanh, complex_tanh_backward, complex_sigmoid, complex_sigmoid_backward

    # Real-valued XOR dataset
    # Use x = (+-1, +-1) for real XOR
    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1]
    ], dtype=np.float64)
    Y = np.array([
        [-1],
        [ 1],
        [ 1],
        [-1]
    ], dtype=np.float64)

    model = Sequential([
        Dense(2, 2, real=True),
        (lambda x: complex_tanh(x, real=True), lambda z, g: complex_tanh_backward(z, g, real=True)),
        Dense(2, 1, real=True),
        (lambda x: complex_sigmoid(x, real=True), lambda z, g: complex_sigmoid_backward(z, g, real=True))
    ])

    print("Training real-valued 2-2-1 on XOR...")
    history = model.fit(X, Y, epochs=2000, lr=0.1, return_history=True)
    out = model.predict(X)
    print("Predictions:", out.round(2).flatten())
    print("Targets:    ", Y.flatten())
    print("Final loss:", history['loss'][-1])

if __name__ == "__main__":
    test_dense()
    test_complex_relu()
    test_complex_sigmoid()
    test_complex_tanh()
    test_modrelu()
    test_jamie_init()
    test_real_xor()
    print("All tests passed.")


def test_real_xor_1000_epochs_plot():
    import numpy as np
    import matplotlib.pyplot as plt
    from cvnn.model import Sequential
    from cvnn.layers import Dense
    from cvnn.activations import complex_tanh, complex_tanh_backward, complex_sigmoid, complex_sigmoid_backward

    # Real-valued XOR dataset
    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1]
    ], dtype=np.float64)
    Y = np.array([
        [-1],
        [ 1],
        [ 1],
        [-1]
    ], dtype=np.float64)

    model = Sequential([
        Dense(2, 2, real=True),
        (lambda x: complex_tanh(x, real=True), lambda z, g: complex_tanh_backward(z, g, real=True)),
        Dense(2, 1, real=True),
        (lambda x: complex_sigmoid(x, real=True), lambda z, g: complex_sigmoid_backward(z, g, real=True))
    ])

    print("Training real-valued 2-2-1 on XOR for 1000 epochs...")
    history = model.fit(X, Y, epochs=1000, lr=0.1, return_history=True)
    out = model.predict(X)
    print("Predictions:", out.round(2).flatten())
    print("Targets:   ", Y.flatten())
    print("Final loss:", history['loss'][-1])

    plt.plot(history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Real-valued XOR Loss over Epochs (1000 epochs)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_real_xor_1000_epochs_plot()


def test_real_xor_multiple_runs(n_runs=20, epochs=1000, loss_threshold=0.1):
    """Run multiple 2-2-1 real-valued XOR trainings and count how many converge (final loss < threshold)."""
    import numpy as np
    from cvnn.model import Sequential
    from cvnn.layers import Dense
    from cvnn.activations import complex_tanh, complex_tanh_backward, complex_sigmoid, complex_sigmoid_backward

    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1]
    ], dtype=np.float64)
    Y = np.array([
        [0],
        [ 1],
        [ 1],
        [0]
    ], dtype=np.float64)

    converged = 0
    losses = []
    for i in range(n_runs):
        model = Sequential([
            Dense(2, 2, real=True),
            (lambda x: complex_tanh(x, real=True), lambda z, g: complex_tanh_backward(z, g, real=True)),
            Dense(2, 1, real=True),
            (lambda x: complex_sigmoid(x, real=True), lambda z, g: complex_sigmoid_backward(z, g, real=True))
        ])
        history = model.fit(X, Y, epochs=epochs, lr=0.1, return_history=True)
        final_loss = history['loss'][-1]
        losses.append(final_loss)
        if final_loss < loss_threshold:
            converged += 1
        print(f"Run {i+1}: Final loss = {final_loss:.4f}{' (converged)' if final_loss < loss_threshold else ''}")
    print(f"\nConverged in {converged}/{n_runs} runs (final loss < {loss_threshold})")
    return losses

if __name__ == "__main__":
    test_real_xor_multiple_runs()


def test_real_xor_multiple_runs_sigmoid(n_runs=20, epochs=1000, loss_threshold=0.1):
    """Run multiple 2-2-1 real-valued XOR trainings with sigmoid hidden activation and count how many converge (final loss < threshold)."""
    import numpy as np
    from cvnn.model import Sequential
    from cvnn.layers import Dense
    from cvnn.activations import complex_sigmoid, complex_sigmoid_backward
    from cvnn.initialisations import complex_normal

    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1]
    ], dtype=np.float64)
    Y = np.array([
        [-1],
        [ 1],
        [ 1],
        [-1]
    ], dtype=np.float64)

    converged = 0
    losses = []
    from cvnn.activations import sigmoid, sigmoid_backward
    for i in range(n_runs):
        model = Sequential([
            Dense(2, 2, real=True, weight_init=lambda shape: np.random.normal(0, 1, size=shape), bias_init=lambda shape: np.random.normal(0, 1, size=shape)),
            (sigmoid, sigmoid_backward),
            Dense(2, 1, real=True, weight_init=lambda shape: np.random.normal(0, 1, size=shape), bias_init=lambda shape: np.random.normal(0, 1, size=shape)),
            (sigmoid, sigmoid_backward)
        ])
        history = model.fit(X, Y, epochs=epochs, lr=0.1, return_history=True)
        final_loss = history['loss'][-1]
        losses.append(final_loss)
        if final_loss < loss_threshold:
            converged += 1
        print(f"[Sigmoid-NormalInit] Run {i+1}: Final loss = {final_loss:.4f}{' (converged)' if final_loss < loss_threshold else ''}")
    print(f"\n[Sigmoid-NormalInit] Converged in {converged}/{n_runs} runs (final loss < {loss_threshold})")
    return losses

if __name__ == "__main__":
    test_real_xor_multiple_runs_sigmoid()
    test_all_inits_and_activations()
