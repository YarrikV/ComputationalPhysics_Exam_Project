import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt

# Global parameters
A = 4		# Depth of potential
a = 0.3		# width of potential
r_e = 4		# offset of maximum
h = 0.01	# integration step
rMax = 26.  # Where potential becomes (almost) zero

b = 2.
R =20.
kr = 1.50716
lower = 1.50710 
upper = 1.50720 


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
	rMax1 = rMax
	rMax2 = np.floor(rMax1 + 0.5 * np.pi / k)

	# Number of steps
	IntStep = int(rMax/h)
	DeltaIntStep = int((rMax2 - rMax1)/h) + 1
	TotalIntStep = IntStep + DeltaIntStep

	# Array to store u, radial distance and psi
	u = np.zeros(TotalIntStep + 1)
	r = np.zeros(TotalIntStep + 1)

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

	return (r, u)


def Gaussian(r, b, R):
	return np.exp(-(r-R)**2/b**2)



def WavePack(lower, upper, t):
	klist = np.linspace(lower, upper, num=1e2)
	phi = np.array([Numerov(0, k**2)[1] for k in klist])
	
	r = Numerov(0,1)[0]
	f = phi*Gaussian(r, b, R)
	
	C = 2/np.pi*np.trapz(f, r)
	expo = np.exp(- 1j*klist**2*t)
	
	g = np.zeros(phi.shape)
	for i in range(len(C)):
		g[i,:] =phi[i,:]*C[i]
	# g =(C*phi.T).T
	f = np.zeros(g.shape, dtype=np.complex)
	for i in range(len(expo)):
		f[i,:] =g[i,:]*expo[i]
	# f = (g.T*expo).T
	
	output = np.trapz(f, klist, axis=0)
	return output

r = Numerov(0,1)[0]
psi = WavePack(lower, upper, 0)
plt.plot(r, np.absolute(psi)**2*1e4, label=r"$t=0\tau$")

plt.yscale('log')

plt.xlim(-1,10)

plt.title("Absolute value resonance part wave packet")

plt.ylabel(r"$|\psi|_R\times 10^4$")
plt.xlabel("Radial distance")

plt.legend()

plt.savefig("ResWavePack.png")

plt.show()