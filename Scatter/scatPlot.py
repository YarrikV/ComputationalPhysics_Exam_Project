import matplotlib.pyplot as plt
from Scatter import main, A
import numpy as np

# Not usable, just shows how we got the cross section and phaseshift figures.

LMin = 0
LMax = 5
energies, deltas, sigmas, psis = main()

fig = plt.figure(2)
axes = fig.add_subplot(2, 1, 1)
axes.set_xlabel(r"Energy $E$ $(fm^{-2})$", fontsize = 15)
axes.set_title(r"Total Cross Section for $l = 0$", fontsize=20)
axes.plot(energies[:320], sigmas[:320])
axes.set_ylabel(r"Cross Section $\sigma(E)$ $(fm^{-2}$)", fontsize=15)
axes.set_xlim(0)


axes = fig.add_subplot(2, 1, 2)
axes.set_xlabel(r"Energy $E$ $(fm^{-2})$", fontsize = 15)
axes.set_title(r"Total Cross Section for $l = 0 - 5$", fontsize=20)
axes.plot(energies[:320], sigmas[:320])
axes.set_ylabel(r"Cross Section $\sigma(E)$ $(fm^{-2}$)", fontsize=15)
axes.set_xlim(0)


axes = fig.add_subplot(2, 1, 1)
axes.set_xlabel(r"Wavenumber $k \ (fm^{-1})$", fontsize= 20)
axes.set_title(r"Phase Shift $\delta_0(k)$ for $l = 0$ and $ A = {:.1f}$".format(A), fontsize= 25)
axes.scatter(np.sqrt(energies), deltas[:,0] / np.pi)
axes.plot(np.sqrt(energies), deltas[:,0] / np.pi, linestyle='--')
axes.set_ylabel(r"$\delta_0(k) \ (\pi)$", fontsize= 20)
axes.set_xlim(0)

A = 2.7
energies, deltas, sigmas, psis = main()


axes = fig.add_subplot(2, 1, 2)
axes.set_xlabel(r"Wavenumber $k \ (fm^{-1})$", fontsize= 20)
axes.set_title(r"Phase Shift $\delta_0(k)$ for $l = 0$ and $ A = {:.1f}$".format(A), fontsize= 25)
axes.scatter(np.sqrt(energies), deltas[:,0] / np.pi)
axes.plot(np.sqrt(energies), deltas[:,0] / np.pi, linestyle='--')
axes.set_ylabel(r"$\delta_0(k) \ (\pi)$", fontsize= 20)
axes.set_xlim(0)

A = 3.3
energies, deltas, sigmas, psis = main()


plt.figure(1)
axes = fig.add_subplot(2, 2, 3)
axes.set_xlabel(r"Wavenumber $k (fm^{-1})$")
axes.set_title(r"Phase Shift $\delta_0(k)$ for $l = 0$ and$ A = {:.3f}$".format(A))
axes.scatter(energies, deltas[:,0] / np.pi)
axes.plot(energies, deltas[:,0] / np.pi, linestyle='--')
axes.set_ylabel(r"$\delta_0(k) \ (\pi)$")
axes.set_xlim(0)

plt.show()
