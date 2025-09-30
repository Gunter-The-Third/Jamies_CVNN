#!/usr/bin/env python3
"""
Test script to verify the fix for matrix dimension issues in CVNN Sequential.fit()
"""

import numpy as np
import sys
import os

# Add the local cvnn directory to path to test the fixed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cvnn'))

from cvnn import Dense, Sequential
from cvnn.activations import complex_sigmoid, complex_sigmoid_backward

def test_fixed_sequential():
    """Test that the fixed Sequential class works without matrix dimension errors."""
    
    print("Testing Fixed CVNN Sequential Class")
    print("=" * 50)
    
    # Test Case 1: Real-valued XOR problem (this was failing before)
    print("\nTest 1: Real-valued XOR problem")
    print("-" * 30)
    
    X_xor = np.array([
        [0.0, 0.0],
        [0.0, 1.0], 
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float64)
    
    Y_xor = np.array([
        [0.0],  # 0 XOR 0 = 0
        [1.0],  # 0 XOR 1 = 1
        [1.0],  # 1 XOR 0 = 1
        [0.0]   # 1 XOR 1 = 0
    ], dtype=np.float64)
    
    try:
        model_real = Sequential([
            Dense(input_dim=2, output_dim=3, real=True, complex=False),
            (np.tanh, lambda x, grad: grad * (1 - np.tanh(x)**2)),
            Dense(input_dim=3, output_dim=1, real=True, complex=False),
            (lambda x: 1/(1+np.exp(-x)), lambda x, grad: grad * (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x))))
        ], real=True)
        
        print("✓ Model created successfully")
        
        losses = model_real.fit(X_xor, Y_xor, epochs=100, lr=0.5, verbose=False)
        print(f"✓ Training completed! Final loss: {losses[-1]:.4f}")
        
        predictions = model_real.forward(X_xor)
        print(f"✓ Predictions: {predictions.flatten().round(3)}")
        print(f"✓ Targets:     {Y_xor.flatten()}")
        
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return False
    
    # Test Case 2: Complex-valued network
    print("\nTest 2: Complex-valued network")
    print("-" * 30)
    
    X_complex = np.array([
        [1+1j],
        [1-1j],
        [-1+1j],
        [-1-1j]
    ], dtype=np.complex128)
    
    Y_complex = np.array([
        [0.0],
        [0.0],
        [1.0],
        [1.0]
    ], dtype=np.complex128)
    
    try:
        model_complex = Sequential([
            Dense(input_dim=1, output_dim=2, complex=True, real=False),
            (complex_sigmoid, complex_sigmoid_backward),
            Dense(input_dim=2, output_dim=1, complex=True, real=False)
        ], real=False)
        
        print("✓ Complex model created successfully")
        
        losses = model_complex.fit(X_complex, Y_complex, epochs=50, lr=0.1, verbose=False)
        print(f"✓ Complex training completed! Final loss: {losses[-1]:.4f}")
        
        predictions = model_complex.forward(X_complex)
        print(f"✓ Complex predictions (real): {predictions.real.flatten().round(3)}")
        
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False
    
    # Test Case 3: Multi-layer with different dimensions
    print("\nTest 3: Multi-layer with varying dimensions")
    print("-" * 30)
    
    X_multi = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=np.float64)
    
    Y_multi = np.array([
        [1.0],
        [0.0]
    ], dtype=np.float64)
    
    try:
        model_multi = Sequential([
            Dense(input_dim=3, output_dim=5, real=True, complex=False),
            (np.tanh, lambda x, grad: grad * (1 - np.tanh(x)**2)),
            Dense(input_dim=5, output_dim=2, real=True, complex=False),
            (np.tanh, lambda x, grad: grad * (1 - np.tanh(x)**2)),
            Dense(input_dim=2, output_dim=1, real=True, complex=False)
        ], real=True)
        
        print("✓ Multi-layer model created successfully")
        
        losses = model_multi.fit(X_multi, Y_multi, epochs=50, lr=0.1, verbose=False)
        print(f"✓ Multi-layer training completed! Final loss: {losses[-1]:.4f}")
        
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! The fix is working correctly.")
    print("✓ Matrix dimension issues have been resolved")
    print("✓ Sequential.fit() now works with activation functions")
    print("✓ Both real and complex-valued networks are working")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_fixed_sequential()
    sys.exit(0 if success else 1)