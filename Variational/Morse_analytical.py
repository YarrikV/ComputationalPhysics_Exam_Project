import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, factorial, gamma

# Construct the n'th analytical wavefunction
def wavefunc_analytical(n, lamb, x_e):
    exponent = 2 * lamb - 2 * n - 1
    N = np.sqrt( factorial(n) * exponent / gamma(2 * lamb - n) )
    C = np.power(2 * lamb, exponent / 2)
    
    return lambda x: N * C * np.exp(- 0.5 * exponent * (x - x_e) - lamb * np.exp(x_e - x)) * eval_genlaguerre(n, exponent, 2 * lamb * np.exp(x_e - x))

# Calculate the analytical energies eigenvalues
def energy_analytical(n, lamb):
    return - np.power(lamb - n - 0.5, 2)