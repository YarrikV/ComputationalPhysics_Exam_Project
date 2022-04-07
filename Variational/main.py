import numpy as np
import matplotlib.pyplot as plt
import Morse as morse
import plot as plot
import Morse_analytical as ana

# Define some global parameters
N = int(15)	# Basis set size
V0 = -50.	# Depth of the well
alpha = 1	# Range
x_e = 0.	# Displacement: w/o loss of generality = 0
lamb = np.sqrt(-V0) / alpha #Only relevant parameter in rescaled situation

# Solve the eigenvalue problem
e, C = morse.solve(N, lamb)

# Energies
E_var = [e for e in e]

n_array = np.array([i for i in range(N)])
E_ana = ana.energy_analytical(n_array, lamb)

# For plotting purposes, defines the range of the plots
x = np.linspace(-2, 10, num=2000)
lastBound = int(np.floor(lamb - 0.5)) # Analytical last bounded stated

# Draw some figures

plot.draw_bound_states(x, E_var, N, V0, alpha)
plot.draw_wavefunction(x, E_var, C, lamb, x_e, lastBound + 1, False, 3)
plot.draw_energy_levels(E_var, E_ana, N, V0, alpha, lastBound)
plot.draw_convergence(E_ana, lamb)

plt.show()
