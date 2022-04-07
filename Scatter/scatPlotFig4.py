from Scatter import main, Morse, h
import matplotlib.pyplot as plt
import numpy as np

# This file works if global parameters are set right, lmin = 0, lmax= 1 at least.

def findZeros(waveSpace, amount=5, start=0):
    '''
    Find where an array crosses zero.
    
    waveSpace = (xs, ys)
        - where len(xs) >= len(ys)
        - Finds x value where y values reach zero.
    
    amount: How many zeroes you want to find.
    start: At which index you start looking for zeros.
    '''
    rs, psis = waveSpace
    rsZero = []
    now = False
    positive = psis[start] > 0
    for i, psi in enumerate(psis[start:]):
        if (psi > 0) and (not positive):
            now = True
        elif (psi < 0) and positive:
            now = True

        if now:
            positive = not positive
            rsZero.append(rs[i+start])
            now = False

        if len(rsZero) > 4:
            break
    return rsZero


def plot_fig4(waveSpace, ks):
    '''Plots figure corresponding to figure 4 in the added paper.'''

    nodes = [[] for _ in range(5)]
    for l in range(1):
        newL = [ener[l] for ener in waveSpace]
        # len(newL[0]) == 6
        for waveSpace in newL:
            zeroRs = findZeros(waveSpace)
            for i, zeroR in enumerate(zeroRs):
                nodes[i].append(zeroR)

    # Hand tailored formatting.
    formatting = ['v-.', 'H--', '^-', 'h-.', '<-.']
    plt.figure(4)

    
    minima, maxima = [], []
    rs, _ = waveSpace
    for k in ks:
        minim, maxim = findZeros((rs, k**2 - Morse(rs)), amount=2)
        minima.append(minim)
        maxima.append(maxim)

    plt.fill_between(ks, maxima, minima, label='Barrier Region',
                  alpha=0.3, color='black', interpolate=True)

    for i, node in enumerate(nodes[1:]):
        plt.plot(ks, node, formatting[i],
                    label=f'R({i+1})')
    plt.legend()
    plt.xlabel(r'$k$')
    plt.ylabel(r'Radial distance ($fm$)')


def uToPsi(waveSpace):
    '''Changes the u 'wave functions' to the 'real' wave function Psi.'''
    rs, us = waveSpace
    us[1] /= h
    us[2:] /= rs[2:]
    return (rs, us)

# NOTE: Understanding the waveSpace variable:
# waveSpace[energielevel][l] = (rs, u)
# psi = u / rs

energies, deltas, sigmas, us = main()
rsPsis = [[uToPsi(l) for l in level] for level in us]
ks = np.sqrt(energies)
plot_fig4(rsPsis, ks)

plt.show()
