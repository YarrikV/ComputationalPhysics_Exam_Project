import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import factorial2
import Morse_analytical as ana

#########
# UNITS #
#########
# Unit of energy: alpha²
# Unit of distance: 1/\alpha
# \hbar^2/2m = 1

def potential(x, lamb):
	'''The Morse Potential.'''
	return - lamb ** 2 * np.exp(-x)*(2. - np.exp(-x))

def basis(n, x):
	'''The basis set / trial functions.'''
	return np.power(1 + x, n + 1)*np.exp(-x**2)


def matrix(nf, lamb):
	'''Produce the Hamiltonian H and overlap matrix S using numpy.'''
	H = [[Ham(n, m, lamb) for n in np.arange(0, nf, 1)] for m in np.arange(0, nf, 1)] # Symmetry could be exploited for efficiency
	S = [[Sp(n, m) for n in np.arange(0, nf, 1)] for m in np.arange(0, nf, 1)]
	return H, S

def spectrum(H, S):
	'''Diagonalize the problem.'''
	e, C = eigh(H, S)
	return e, C

def Sp(n, m):
	'''Overlap Integrals.'''
	x = np.linspace(-10, 10, num=1e4)
	f = basis(m,x)*basis(n,x)
	return np.trapz(f,x)


def V(n, m, lamb):
	'''Calculate potential energy numerically.'''
	x = np.linspace(-10, 10, num=1e4)
	fn = basis(n, x)
	fm = basis(m, x)
	V = potential(x, lamb)
	return np.trapz(fn*V*fm, x)

def T(n, m):
	'''Kinetic Energy Integrals.'''
	x = np.linspace(-10, 10, num=1e4)

	second_deriv = np.power(1 + x, m - 1)
	second_deriv *= np.exp(- x ** 2) * (m ** 2 - 4 * m * x ** 2 - 4 * m * x + m + 4 * x ** 4 + 8 * x ** 3 - 2 * x ** 2 - 8 * x - 2)
	f = -basis(n, x)*second_deriv
	
	return np.trapz(f, x) 

def Ham(n, m, lamb):
	'''Hamiltonian.'''
	return V(n, m, lamb) + T(n,m)

def normalise(wavefunc, x):
	'''Normalisation of the wavefunction.'''
	N = np.sqrt(np.trapz(wavefunc * wavefunc, x))

	return wavefunc / N


def wavefunc_n(n, C, x):
	'''Extract the nth wavefunction.'''
	wavefunc = 0
	for i in range(len(C)):
		wavefunc += C[i][n] * basis(i, x)
	
	return normalise(wavefunc, x)

def solve(N, lamb):
	'''Solve the eigenvalue problem with basis set of size N.'''
	return spectrum(*matrix(N, lamb))
