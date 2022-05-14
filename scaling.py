#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:19:36 2022

@author: jp
"""


# Standard library imports:
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# Local imports:
from setup import setup, load_data, get_atom_numbers_all_shots
from plot_colors import get_plot_colors
from post_selection import set_ctrl_vals_for_ps
from variance import variance, plot_variance
                            

""" ---------- INPUT ---------- """
USE_ATOM_NUMBER_CALIB_UJ = False
REL_FLUCT_TARGETS = [0.42, 3, 6, 8]
ETA = 0.53

#%% Execution:

if __name__ == '__main__':
    
    data_basepath, figure_savepath, lattice_atom_number_calibration = setup()
    recentered_data, uj_vals = load_data(data_basepath)
    atom_numbers_all_shots = get_atom_numbers_all_shots(
                                recentered_data,
                                uj_vals
                                )
    
        
    
#%% Scaling

# def scaling(CTRL_VAL_SHIFTS):
    
CTRL_VAL_SHIFT = 500
    
    
CTRL_VAL_SHIFTS = np.linspace(-CTRL_VAL_SHIFT, CTRL_VAL_SHIFT, 100)
    
# Prepare data containers:
relative_fluctuations_sc = pd.DataFrame(data=None, index=CTRL_VAL_SHIFTS, columns=REL_FLUCT_TARGETS)
relative_fluctuations_error_sc = pd.DataFrame(data=None, index=CTRL_VAL_SHIFTS, columns=REL_FLUCT_TARGETS)
ps_atom_numbers_sc = dict.fromkeys(CTRL_VAL_SHIFTS)
fluct_std_perc_sc = dict.fromkeys(CTRL_VAL_SHIFTS)

for ctrl_val_shift in tqdm(CTRL_VAL_SHIFTS, desc='Scaling'):
    
    ps_ctrl_vals = set_ctrl_vals_for_ps(
                    USE_ATOM_NUMBER_CALIB_UJ,
                    ctrl_val_shift,
                    lattice_atom_number_calibration,
                    uj_vals
                    )
    (
     ps_atom_numbers_sc[ctrl_val_shift],
     fluct_std_perc_sc[ctrl_val_shift],
     relative_fluctuations,
     relative_fluctuations_error,
     sts,
     sts_error
     ) = variance(
         uj_vals,
         atom_numbers_all_shots,
         recentered_data,
         ps_ctrl_vals,
         REL_FLUCT_TARGETS,
         plot_ps=False
         )
    relative_fluctuations_sc.at[ctrl_val_shift] = relative_fluctuations.loc[24]
    relative_fluctuations_error_sc.at[ctrl_val_shift] = relative_fluctuations_error.loc[24]


#%%    
    
plt.figure(figsize=(19, 9))
plot_colors = get_plot_colors(
        'qualitative',
        max([len(REL_FLUCT_TARGETS), 3]),
        name='Set1'
        )

popt = pd.DataFrame(data=None, index=REL_FLUCT_TARGETS, columns=['exp', 'offset'])
# pcov = pd.DataFrame(data=None, index=REL_FLUCT_TARGETS, columns=['slope', 'offset'])
atom_numbers = pd.Series(data=None, index=REL_FLUCT_TARGETS, dtype=object)

def ftn_fctn(x, a, b):
    return x**(1+a) + b

from scipy.optimize import curve_fit

for idx, rel_fluct_target in enumerate(REL_FLUCT_TARGETS):
    
    atom_numbers[rel_fluct_target] = [ps_atom_numbers_sc[ctrl_val_shift][rel_fluct_target][24].mean() / ETA for ctrl_val_shift in CTRL_VAL_SHIFTS]
    
    plt.errorbar(
        atom_numbers[rel_fluct_target],
        relative_fluctuations_sc[rel_fluct_target],
        yerr=relative_fluctuations_error_sc[rel_fluct_target],
        color=plot_colors[idx],
        label=(r'$\frac{\Delta N}{N} = $'
                         + f'{fluct_std_perc_sc[ctrl_val_shift][rel_fluct_target]:.1f}%  - '
                         + 'Ground State Fluct.'),
        marker='o',
        markersize=5,
        linewidth=0,
        elinewidth=1.5,
        capsize=4.0,
        zorder=1/(idx+1),
        )
    
    # Fit:
    popt.at[rel_fluct_target], _ = curve_fit(
        ftn_fctn,
        np.array(atom_numbers[rel_fluct_target])[np.where(relative_fluctuations_sc[rel_fluct_target].notna())[0]],
        relative_fluctuations_sc[rel_fluct_target].dropna()
        )    
    
fit_plot_atom_numbers = np.linspace(min([min(atom_numbers[i]) for i in REL_FLUCT_TARGETS]), max([max(atom_numbers[i]) for i in REL_FLUCT_TARGETS]), 100)
plt.plot(
    fit_plot_atom_numbers,
    [ftn_fctn(i, popt.exp.mean(), popt.offset.mean()) for i in fit_plot_atom_numbers],
    color='k',
    label='Fit: '+r'$\frac{\Delta N_0^2}{N} \propto N^{1 + \gamma}; \gamma_{\mathrm{fit}} = $'+f'{popt.exp.mean():.2}'+r'$\ ; \gamma_{\mathrm{theo}} = -\frac{2}{3}$'
    )
plt.xlabel(r'$N_0$')
plt.ylabel(r'$\Delta N_{0}^2|_{\frac{U}{J}=24}\ /\ N$')
plt.title('Scaling of the ground state occupation fluctuations with the atom number at U/J = 24')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

    
# scaling(CTRL_VAL_SHIFTS)

                 