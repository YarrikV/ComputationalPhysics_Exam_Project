import numpy as np
import matplotlib.pyplot as plt
import Morse as morse
import Morse_analytical as ana
# Some definitions for plotting figures

# Draw the Morse potential with the variational bound states
def draw_bound_states(x, E_var, N, V0, alpha):
	plt.figure(1)
	lamb = np.sqrt(-V0) / alpha
	plt.plot(x, morse.potential(x, lamb))

	# Bound state: energy < 0
	for e in E_var:
		if e<0:
			plt.axhline(y=e, color='black', linestyle='--', alpha=0.5)
	
	plt.xlim(-1.5,10)
	plt.ylim(-60, 10)
	plt.xlabel(r"Separation $x$ ($\alpha^{-1}$)", fontsize=12)
	plt.ylabel(r"Energy $E$ ($\alpha^2$)",fontsize=12)
	plt.title(r"Morse Potential with Bound States for $N_b$ = {}, $V_0$ = {}, $\alpha$ = {}".format(N, V0, alpha),fontsize=20)

	pass

# Draw some probability densities up to lastIndex
# Remark: analytical bound may be an unbound state in the variational problem
# m: defines mxm grid for the subplots
def draw_wavefunction(x, E_var, C, lamb, x_e, lastIndex, density=True, m=3):
	fig = plt.figure(2)
	for n in range(lastIndex):

		# Construct variational and analytical wavefunctions
		psi = morse.wavefunc_n(n, C, x)
		psi_ana = ana.wavefunc_analytical(n, lamb, x_e)(x)

		# Check if bound state
		if E_var[n] < 0:
			# Search for the well area
			difference = morse.potential(x, lamb) - E_var[n]
			intersections = []
			for i in range(1, len(difference)):
				if np.sign(difference[i]) != np.sign(difference[i - 1]):
					intersections.append(-2 + (2 * i - 1) / 2 * 12 / 2000)
		else:
			# Arbitrary values for unbound states
			intersections = [-.7, 4.5]
		
		axes = fig.add_subplot(m, m, n + 1)
		axes.set_xlabel(r"Separation $x$ ($\alpha^{-1}$)")
		axes.set_title(r"{}th Wavefunction".format(n))

		# Plot the probability wavefunctions or the densities
		if density:
			axes.plot(x, psi ** 2, label="variational")
			axes.plot(x, psi_ana ** 2, label="analytical", linestyle="--")
			axes.set_ylabel(r"$|\psi(x)|^2$")
		else:
			axes.plot(x, psi, label="variational")
			axes.plot(x, psi_ana, label="analytical", linestyle="--")
			axes.set_ylabel(r"$\psi(x)$")
		axes.legend()
		axes.axvspan(intersections[0], intersections[1], color='grey', alpha=0.25)
	
	plt.tight_layout()
	fig.set_size_inches((15, 11), forward=False)

	pass

# Draw the energy levels up to lastIndex: variational and analytical
def draw_energy_levels(E_var, E_ana, N, V0, alpha, lastIndex):
	plt.figure(3)

	plt.axhline(y=E_var[0], xmin=0.3, xmax=0.7, color='b', label="variational")
	plt.axhline(y=E_ana[0], xmin=0.3, xmax=0.7, color='r', label='analytical')
	plt.text(0.27, E_var[0], "n = {}".format(0))
	plt.text(0.71, E_ana[0], "n = {}".format(0))

	for i in range(1, lastIndex):
		plt.axhline(y=E_var[i], xmin=0.3, xmax=0.7, color='b')
		plt.axhline(y=E_ana[i], xmin=0.3, xmax=0.7, color='r')
		plt.text(0.27, E_var[i], "n = {}".format(i))
		plt.text(0.71, E_ana[i], "n = {}".format(i))

	plt.xlim(0, 1)
	plt.xticks([])
	plt.ylabel(r"Energy $E$ ($\alpha^2$)",fontsize=12)
	plt.title(r"Energy of the Bound States for $N_b$ = {}, $V_0$ = {}, $\alpha$ = {}".format(N, V0, alpha),fontsize=20)
	plt.legend()

	pass

# Draw convergence of the ground state energy with the basis set size
def draw_convergence(E_ana, lamb):

	plt.figure(4)

	for i in range(1, 15):
		plt.scatter(i, morse.spectrum(*morse.matrix(i, lamb))[0][0], color='black', marker='*')

	plt.xlabel(r"Amount of Basis Functions $N_b$",fontsize=12)
	plt.ylabel(r"Ground State Energy $E_{gs}$ ($\alpha^2$)",fontsize=12)
	plt.title("Convergence of Ground State Energy with Size of Basis Set",fontsize=20)
	plt.axhline(E_ana[0], linestyle='--', color='black')

	pass
