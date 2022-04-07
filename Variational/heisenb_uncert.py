from Morse import wavefunc_n
from main import alpha, C, x
from numpy import exp, sqrt, trapz
import numpy as np
import matplotlib.pyplot as plt

def expectedValue(f, N, operatingOnPsi=False): 
    '''Calculates expected value of 'operator'.
    operatingOnPsi True when the operator actually has an effect
                        on the wavefunction. (eg. momentum op.)
                   False when the wavefunction can just be squared
                         (assumed real wavefunction). (eg. position op.) 
    '''
    if operatingOnPsi:
        val = trapz(wavefunc_n(N, C, x) * f, x)
    else:
        val = trapz(wavefunc_n(N, C, x)**2 * f, x)
    return val


def heisUncert(n, verbose=True):
    '''Calculates DxDp for the Morse Potential.
    Compares Morse potential results with the HO case.

    verbose (Bool): If true, writes output.
    returns Dx, Dp* , DxDp_Morse*, DxDp_HO*
        * in unit hbar
    '''
    dx, dp, heisM, heisHO = 0, 0, 0, 0
    expX = expectedValue(x, n)
    expX2 = expectedValue([elem**2 for elem in x], n)

    expP = expectedValue(P(n), n, operatingOnPsi=True)
    expP2 = expectedValue(P2(n), n, operatingOnPsi=True)

    dx, dp = sqrt(abs((expX2 - expX**2))), sqrt(abs((expP2 + expP**2)))
    heisM = dx * dp
    heisHO = n + 0.5
    if verbose:
        print(f'For energy level n={n}:')
        print(f'Morse: {heisM:.2f}, HO: {heisHO:.2f}')
        print(f'Difference: {heisM - heisHO:.2f}')
        print()
    return dx, dp, heisM, heisHO


P = lambda n : (((1 + x)**n *(1 + n - 2 * x - 2 * x**2)) * np.exp(- x ** 2)) / alpha
P2 = lambda n : - (np.power(1 + x, n - 1) * (np.exp(- x ** 2) * (n ** 2 - 4 * n * x ** 2 - 4 * n * x + n + 4 * x ** 4 + 8 * x ** 3 - 2 * x ** 2 - 8 * x - 2))) / alpha**2

x = np.linspace(-10, 20, num=2000)


