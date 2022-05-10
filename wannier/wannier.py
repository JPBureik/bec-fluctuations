#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:52:44 2022

@author: jp
"""

import numpy as np
import os
from matplotlib import pyplot as plt

currentfolder = os.getcwd()


lattice = 5.85
#if already calculated load the wannier from the correct repository
kint,wk = np.load('/home/jp/Documents/prog/work/bec-fluctuations/wannier/wk_s=10.2.npy')
# kint=kint
# wint = interp1d(kint,wk, kind='cubic')

f1, ax = plt.subplots(1,1,figsize = (8,4.9))
# ax.plot(kx, nkx/(1*max(nkx)),  
#           alpha = 1,
#           linewidth = 3,
#           label = r'Exp. $n(k,0,0)$')
# ax.plot(kx, nky/(1*max(nky)),  
#           alpha = 1,
#           linewidth = 3,
#           label = r'Exp. $n(0,k,0)$')
# ax.plot(kx, nkz/(1*max(nkz)),  
#           alpha = 1,
#           linewidth = 3,
#           label = r'Exp. $n(0,0,k)$')

ax.plot(kint, wk,
          color =  '#092e86',
          linestyle = 'dashed',
          alpha = 1,
          linewidth = 2,
          label = r'Wannier')

ax.set(xlim = (-1.5,1.5),
        xlabel = r'$k\,[k_{latt}]$', 
        ylabel = r'[arb. units]')
ax.grid(linestyle = 'dashed')
ax.legend(loc = 'upper left',frameon = True)
f1.tight_layout()    

# interpolation of the wannier function    
# wk_interp = interp1d(kint, wk, kind='cubic')
# plt.subplots(1,1,figsize = (8,4.9))
# plt.plot(kint, wk, '-', kx, wk_interp(kx), 'o')

