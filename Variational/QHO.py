import numpy as np
import matplotlib.pyplot as plt
from main import V0, alpha
from scipy.optimize import curve_fit

# QHO for fitting
def QHO(x, k):
    return 0.5 * k * x ** 2 + V0

# Taylor expansion of Morse
def Taylor(x, alpha, V0):
    k = -2 * V0 * alpha ** 2
    return 0.5 * k * x ** 2 + V0

# Morse potential
def Morse(x, alpha, V0):
    return V0 * np.exp(-alpha * x) * (2 - np.exp(-alpha * x))

xfitlower, xfitupper = -.2, .95

x = np.linspace(-1., 2., num=2000)
xfit = np.linspace(xfitlower, xfitupper, num=2000)

plt.axvspan(xfitlower, xfitupper, alpha=0.5, color='gray')

# Potentials
morse = Morse(x, alpha, V0)
taylor = Taylor(x, alpha, V0)

# Fitting of QHO to Morse
popt, _ = curve_fit(QHO, xfit, morse)
print(*popt)

# Plot the result
plt.figure(1)
plt.title("Harmonic Oscillator Fit")
plt.xlabel(r"Separation x")
plt.ylabel(r"Potential Energy $E_0$")
plt.plot(x, morse, label="Morse")
plt.plot(x, taylor, label="Taylor: k = {:.2f}".format(-2 * V0 * alpha), linestyle="--", color='red')
plt.plot(x, QHO(x, *popt), label="QHO: k = {:.2f}".format(*popt), color='black')

plt.ylim(-60,10)
plt.xlim(-1.,2.)

plt.legend()
plt.show()