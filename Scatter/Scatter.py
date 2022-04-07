import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt

# units
# all units are converted to fm, as done in the paper

# Global parameters
A = 6		# Depth of potential (fm^-2)
a = 0.3		# width of potential (fm^-1)
r_e = 4		# offset of maximum (fm)
h = 0.005	# integration step
EStart = 0.1 # Start of energy values to be calculated
EStop = 16  # End of energy values to be calculated
ESteps = 500# Number of energy steps
LMin = 0	# Minimum l
LMax = 0	# Maximum l
rMax = 10.  # Where potential becomes (almost) zero



def Morse(r):
	''' Morse Potential.'''
	return A * np.exp(- a * (r - r_e)) * (2 - np.exp(- a * (r - r_e)))

def F(r, l, E):
	'''Help function for Numerov method.'''
	return l * (l + 1) / (r ** 2) + Morse(r) - E

# Numerov algorithm
def Numerov(l, E):
	'''Integrates the Schrodinger equation using the Numerov Method. 
    Psi''(R_I) = FArr(I) Psi(R_I)

	rMax: Limit for integration.
		  Big enough so if one includes r > rMAx, the
		  integration doesn't change much.	
    '''
	k = np.sqrt(E)

	# Integration points for phase shift
	rMax1 = rMax
	rMax2 = np.floor(rMax1 + 0.5 * np.pi / k)

	# Number of steps
	IntStep = int(rMax/h)
	DeltaIntStep = int((rMax2 - rMax1)/h) + 1
	TotalIntStep = IntStep + DeltaIntStep

	# Array to store u, radial distance and psi
	u = np.zeros(TotalIntStep + 1)
	r = np.zeros(TotalIntStep + 1)

	# First 2 Values
	u[1] = h**(l+1.)

	w_h = 0
	w_0 = (h**(l+1.)) * (1. - (h**2/12)*F(h, l, E))

	# Numerov to determine the wavefunction
	for step in range(2, IntStep):

		radius = step * h
		wh = 2*w_0 - w_h + h**2*F((step-1)*h, l, E)*u[step-1]

		output = wh/(1 - (h**2/12)*F(step*h, l, E))
		r[step] = radius
		u[step] = output
	
		w_h = w_0
		w_0 = wh

	# Numerov behind the max distance

	for step in range(IntStep, TotalIntStep + 1):
		
		radius = step * h
		wh = 2*w_0 - w_h + h**2*F((step-1)*h, l, E)*u[step-1]

		output = wh/(1 - (h**2/12)*F(step*h, l, E))
		r[step] = radius
		u[step] = output

		w_h = w_0
		w_0 = wh
	
	# Store and return these values for the phase shift
	r12 = (r[IntStep], r[TotalIntStep])
	u12 = (u[IntStep], u[TotalIntStep])

	return (r, u, r12, u12)
	

def phase_shift(r12, u12, E, l):
	'''Calculates the phaseShift.'''
	k = np.sqrt(E)
	r1, r2 = r12
	u1, u2 = u12
	
	K = r1 / r2 * (u2 / u1)

	numerator = K * spherical_jn(l, k * r1) - spherical_jn(l, k * r2)
	denominator = K * spherical_yn(l, k * r1) - spherical_yn(l, k * r2)

	return np.arctan(numerator / denominator) 

def Cross_section(E, lArray, delta):
	'''Calculate the total cross section according to 2.8 in Thijssen.'''
	sigma = 0
	for i, l in enumerate(lArray):
		sigma += (2*l+1) * np.sin(delta[i]) ** 2
	sigma *= 4 * np.pi / E

	return sigma

def main():
	'''Main Program returning the required quantities'''
	EnergyRange = np.linspace(EStart, EStop, ESteps) # One array of all energies
	LRange = np.arange(LMin, LMax + 1, dtype=int) # One array of all l

	PhaseShifts = np.zeros((ESteps, LMax-LMin + 1)) # A tensor with (i, j) = (E, l)
	crossSection = np.zeros(len(EnergyRange)) # Cross section for certain energy

	waveFunc = [[0 for i in range(LMax-LMin + 1)] for j in range(ESteps)] # Wavespace consisting of (r, u) according to (i, j) = (E, l)

	for i, E in enumerate(EnergyRange):
		for j, l in enumerate(LRange):
			r, u, rMax, uMax = Numerov(l, E)

			# Normalisation and convert to radial wavefunction
			u /= np.sqrt(np.trapz(u * u, r))

			waveFunc[i][j] = (r, u)
			PhaseShifts[i][j] = phase_shift(rMax, uMax, E, l)
		
		crossSection[i] = Cross_section(E, LRange, PhaseShifts[i,:]) 

			
	return EnergyRange, PhaseShifts, crossSection, waveFunc	

def plot_dvsE(EnergyRange, PhaseShifts, l, A):
	'''Plot the phase shift of a certain l'''
	plt.figure(1)
	plt.scatter(np.sqrt(EnergyRange), PhaseShifts[:,l] / np.pi)
	plt.plot(np.sqrt(EnergyRange), PhaseShifts[:,l] / np.pi, linestyle='--', label="A = {:.3f}".format(A))
	plt.xlim(0)
	plt.title(r"Phase Shift $\delta_0(k)$ for $l = 0$", fontsize = 25)
	plt.xlabel(r"Wavenumber $k \ (fm^{-1})$", fontsize = 20)
	plt.ylabel(r"$\delta_0(k) \ (\pi)$", fontsize = 20)
	plt.legend()
	
	pass

def plot_sigma(EnergyRange, CrossSection):
	'''Plot total cross section for all calculated l in function of energy'''
	plt.figure(2)
	plt.plot(EnergyRange, CrossSection)
	plt.title(r"Total Cross Section for different Energies", fontsize= 25)
	plt.xlabel(r"Energy $E$ $(fm^{-2})$", fontsize = 20)
	plt.ylabel(r"Total Cross Section $\sigma(E)$ $(fm^{-2}$)", fontsize=20)
	plt.xlim(0)

	pass

def plot_wavefunc(waveSpace, EnergyRange, l):
	'''Plot some wavefunction u within a certain energy range for a certain l'''
	plt.figure(3)
	for i, E in enumerate(EnergyRange):
		k = np.sqrt(E)
		radii, u = waveSpace[i][l]
		plt.plot(radii[2:], u[2:], label="k = {:.2f}".format(k))
	plt.xlim(0)
	plt.axhline(0, color='black')
	plt.title(r"Normalised Wavefunction $u_0(r)$ near resonant energies for $l = 0$", fontsize = 25)
	plt.xlabel(r"Radial distance $r$ ($fm$)", fontsize = 20)
	plt.ylabel(r"Wavefunction $u_0(r)$", fontsize = 20)
	plt.legend()

	pass

if __name__ == '__main__':
	# Run the main program and store all results
	energies, deltas, sigmas, psis = main()

	# Make some plots
	plot_dvsE(energies, deltas, 0, A)
	plot_sigma(energies, sigmas)
	
	# Only works good for several energies around the resonance (eg. 6 for energies around 5.65).
	plot_wavefunc(psis, energies, 0)

	#plt.tight_layout()
	plt.show()

