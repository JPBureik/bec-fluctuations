#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:37:24 2022

@author: jp
"""

import numpy as np
from matplotlib import pyplot as plt

def theoretical_fluctuations(N, T):
    d = 775e-9  # nm
    R_BEC = 23 * d  # From PRL Cayla 2020
    V_BEC = (4/3)* np.pi * R_BEC**3
    R_Mott = 27.5 * d  # From Carcy PRX 2019
    V_Mott = (4/3)* np.pi * R_Mott**3
    
    
    
    
    
    n_0_at_0 = N / (0.1 * V_BEC)  # At zero T
    h = 6.62607015 * 10**(-34)  # kg * m**2 / s
    m_He = 6.6464731 * 10**(-27)  # kg
    k_B = 1.380649 * 10**(-23)  # kg * m**2 / (K * s**2)
    
    J = h * 450  # Hz
    
    T_min = 2 * J  # from PRL Carcy 2021
    
    T_c = 5.9 * J / k_B  # K, = 5.9 J/k_B
    
    def V(T):
        return T * (V_Mott - V_BEC)/(T_c - T_min)
    
    def n(N, T):
        return N / V(T)
    
    def L(T):
        return V(T)**(1/3)


    
    def lambda_T(T):
        return h * 1/np.sqrt(2 * np.pi * m_He * k_B * T)
    
    if T < T_c:
    
        return 0.8375 * (n_0_at_0 / n(N, T))**2 * (L(T) / lambda_T(T))**4
    
    else:
        
        return 0

plt.figure()

N_sample = [5e3, 5e5]

fluct = dict.fromkeys(N_sample)

for N in N_sample:    
    
    samples = 100
    
    fluct[N] = np.zeros(samples)
    
    T_sample = np.linspace(0.5e-7, 1.5e-7, samples)
    
    for i, T in enumerate(T_sample):
        fluct[N][i] = theoretical_fluctuations(N, T)
        
    
    plt.plot(
        T_sample,
        fluct[N],
        linestyle='--'
        )
plt.show()
    