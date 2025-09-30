import numpy as np
from cvnn.model import Sequential
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


# XOR datasets: real x = (+-1, +-1), y = -1/1; complex x = (+-1)+-(1j), y = -1/1
# Real XOR: x = (+-1, +-1), y = 0/1
X_real = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
], dtype=np.float64)
Y_real = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float64)

# For complex, x is (4,1) with (+-1)+-(1j)
# Complex XOR: x = (+-1)+-(1j), y = 0/1
X_complex = np.array([
    [-1-1j],
    [-1+1j],
    [ 1-1j],
    [ 1+1j]
], dtype=np.complex128)
Y_complex = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.complex128)

# Model builders
def make_single_neuron_complex(weight_init, bias_init, activation, activation_deriv):
    return Sequential([
        Dense(2, 1, weight_init=weight_init, bias_init=bias_init),
        (activation, activation_deriv)
    ])

def make_2_2_1_real(weight_init, bias_init, activation, activation_deriv):
    return Sequential([
        Dense(2, 2, weight_init=weight_init, bias_init=bias_init, real=True),
        (activation, activation_deriv),
        Dense(2, 1, weight_init=weight_init, bias_init=bias_init, real=True),
        (activation, activation_deriv)
    ])

# Experiment settings
init_methods_real = [normal, glorot_uniform, he_normal, zeros, ones]
init_methods_complex = [complex_normal, complex_glorot_uniform, complex_he_normal, complex_zeros, complex_ones]
activations_real = [
    (relu, relu_backward, 'relu'),
    (sigmoid, sigmoid_backward, 'sigmoid'),
    (tanh, tanh_backward, 'tanh')
]
activations_complex = [
    (complex_relu, complex_relu_backward, 'complex_relu'),
    (complex_sigmoid, complex_sigmoid_backward, 'complex_sigmoid'),
    (complex_tanh, complex_tanh_backward, 'complex_tanh')
]
lrs = [0.01, 0.05, 0.1]

# Run experiment
def run_experiment():
    results = []
    for w_init in init_methods_real:
        for b_init in init_methods_real:
            for act, act_b, act_name in activations_real:
                for lr in lrs:
                    converged = 0
                    for _ in range(100):
                        model = make_2_2_1_real(w_init, b_init, act, act_b)
                        history = model.fit(X_real, Y_real, epochs=1000, lr=lr, return_history=True)
                        final_loss = history['loss'][-1]
                        if final_loss < 0.1:
                            converged += 1
                    results.append(('rvnn', w_init.__name__, b_init.__name__, act_name, lr, converged))
                    print(f"RVNN | w_init={w_init.__name__}, b_init={b_init.__name__}, act={act_name}, lr={lr} -> {converged}/100")
    for w_init in init_methods_complex:
        for b_init in init_methods_complex:
            for act, act_b, act_name in activations_complex:
                for lr in lrs:
                    converged = 0
                    for _ in range(100):
                        model = make_single_neuron_complex(w_init, b_init, act, act_b)
                        history = model.fit(X_complex, Y_complex, epochs=1000, lr=lr, return_history=True)
                        final_loss = history['loss'][-1]
                        if final_loss < 0.1:
                            converged += 1
                    results.append(('cvnn', w_init.__name__, b_init.__name__, act_name, lr, converged))
                    print(f"CVNN | w_init={w_init.__name__}, b_init={b_init.__name__}, act={act_name}, lr={lr} -> {converged}/100")
    # Save results
    import csv
    with open('motivation_xor_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'weight_init', 'bias_init', 'activation', 'lr', 'converged'])
        writer.writerows(results)
    print('Results saved to motivation_xor_results.csv')

if __name__ == "__main__":
    run_experiment()
