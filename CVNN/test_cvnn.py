import numpy as np
from cvnn.layers import ComplexDense
from cvnn.activations import (
    complex_relu, complex_sigmoid, complex_tanh, modrelu,
    complex_glorot_uniform, complex_he_normal,
    complex_zeros, complex_ones, complex_normal,
    jamie
)
def test_jamie_init():
    # Test jamie initialisation for phase constraint
    w = jamie((1000,))
    phases = np.angle(w)
    # Should be close to pi/4 or 5pi/4
    close_to_pi4 = np.abs(np.angle(np.exp(1j*(phases - np.pi/4)))) < 0.1
    close_to_5pi4 = np.abs(np.angle(np.exp(1j*(phases - 5*np.pi/4)))) < 0.1
    assert np.all(close_to_pi4 | close_to_5pi4), "Not all phases are close to pi/4 or 5pi/4"
    print("jamie initialisation passed phase constraint test.")

def test_complex_dense():
    # Test with default init
    layer = ComplexDense(input_dim=3, output_dim=2)
    x = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
    out = layer.forward(x)
    print("Output of ComplexDense layer (default init):")
    print(out)
    assert out.shape == (5, 2), f"Expected output shape (5, 2), got {out.shape}"

    # Test with Glorot uniform init
    layer2 = ComplexDense(3, 2, weight_init=complex_glorot_uniform, bias_init=complex_glorot_uniform)
    out2 = layer2.forward(x)
    print("Output of ComplexDense layer (Glorot init):")
    print(out2)
    assert out2.shape == (5, 2)

    # Test with He normal init
    layer3 = ComplexDense(3, 2, weight_init=complex_he_normal, bias_init=complex_he_normal)
    out3 = layer3.forward(x)
    print("Output of ComplexDense layer (He normal init):")
    print(out3)
    assert out3.shape == (5, 2)

    # Test with zeros init
    layer4 = ComplexDense(3, 2, weight_init=complex_zeros, bias_init=complex_zeros)
    out4 = layer4.forward(x)
    print("Output of ComplexDense layer (zeros init):")
    print(out4)
    assert np.allclose(out4, 0)

    # Test with ones init
    layer5 = ComplexDense(3, 2, weight_init=complex_ones, bias_init=complex_ones)
    out5 = layer5.forward(x)
    print("Output of ComplexDense layer (ones init):")
    print(out5)
    # Output should be sum of input along last axis + 1 for each output
    expected = np.sum(x, axis=1, keepdims=True) * np.ones((1,2)) + 1
    assert out5.shape == (5, 2)

    # Test with normal init
    layer6 = ComplexDense(3, 2, weight_init=complex_normal, bias_init=complex_normal)
    out6 = layer6.forward(x)
    print("Output of ComplexDense layer (normal init):")
    print(out6)
    assert out6.shape == (5, 2)

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

if __name__ == "__main__":
    test_complex_dense()
    test_complex_relu()
    test_complex_sigmoid()
    test_complex_tanh()
    test_modrelu()
    test_jamie_init()
    print("All tests passed.")
