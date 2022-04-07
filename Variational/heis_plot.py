import matplotlib.pyplot as plt
from heisenb_uncert import heisUncert
from main import E_var as E

heisMs, heisHOs = [],[]
for i in range(len(E)):
    _,_,heisM,heisHO = heisUncert(i, verbose=True)
    heisMs.append(heisM)
    heisHOs.append(heisHO)


n = [n for n in range(len(E))]

plt.figure()
plt.plot(n, heisMs, label=r"Morse")
plt.plot(n, heisHOs, label=r"HO")
plt.plot(n, E, label="Energies")
plt.legend()
plt.show()
