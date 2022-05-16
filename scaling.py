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
from mcpmeas.helper_functions import multiproc_df

""" ---------- INPUT ---------- """
USE_ATOM_NUMBER_CALIB_UJ = False
REL_FLUCT_TARGETS = [5, 20]
ETA = 0.53
UJ_SCALING = 24

#%% Execution:

if __name__ == '__main__':
    
    data_basepath, figure_savepath, lattice_atom_number_calibration = setup()
    recentered_data, uj_vals = load_data(data_basepath)
    atom_numbers_all_shots = get_atom_numbers_all_shots(
                                recentered_data,
                                uj_vals
                                )
    
        
    
#%% Scaling

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

CTRL_VAL_SHIFT_RANGE = 1#e3
    
CTRL_VAL_SHIFTS = np.linspace(-CTRL_VAL_SHIFT_RANGE, CTRL_VAL_SHIFT_RANGE, 3)#500)

df_cols = pd.MultiIndex.from_product([uj_vals, REL_FLUCT_TARGETS, ['Value', 'Error', 'ps_at_nb', 'fluct_std_perc_sc']])
    
# Prepare data containers:
relative_fluctuations_sc = pd.DataFrame(data=None, index=CTRL_VAL_SHIFTS, columns=df_cols)

# for ctrl_val_shift in tqdm(CTRL_VAL_SHIFTS, desc='Scaling'):
    
def scaling(relative_fluctuations_sc):
    
    for ctrl_val_shift in relative_fluctuations_sc.index:
    
    
        ps_ctrl_vals = set_ctrl_vals_for_ps(
                        USE_ATOM_NUMBER_CALIB_UJ,
                        ctrl_val_shift,
                        lattice_atom_number_calibration,
                        uj_vals
                        )
        var_result = variance(
             uj_vals,
             atom_numbers_all_shots,
             recentered_data,
             ps_ctrl_vals,
             REL_FLUCT_TARGETS,
             plot_ps=False
             )
        for uj in uj_vals:
            for rel_fluct_target in REL_FLUCT_TARGETS:
                relative_fluctuations_sc.loc[ctrl_val_shift][uj][rel_fluct_target]['ps_at_nb'] = var_result[0][rel_fluct_target][uj]
                if (var_result[2][rel_fluct_target][uj] > 0.1 and var_result[3][rel_fluct_target][uj] < 0.25):
                    relative_fluctuations_sc.loc[ctrl_val_shift][uj][rel_fluct_target].at['fluct_std_perc_sc'] = var_result[1][rel_fluct_target]
                    relative_fluctuations_sc.loc[ctrl_val_shift][uj][rel_fluct_target].at['Value'] = var_result[2][rel_fluct_target][uj]
                    relative_fluctuations_sc.loc[ctrl_val_shift][uj][rel_fluct_target].at['Error'] = var_result[3][rel_fluct_target][uj]
         
    return relative_fluctuations_sc
#%%
relative_fluctuations_sc = multiproc_df(relative_fluctuations_sc, scaling)

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
    
    atom_numbers[rel_fluct_target] = [ps_atom_numbers_sc[ctrl_val_shift][rel_fluct_target][UJ_SCALING].mean() / ETA for ctrl_val_shift in CTRL_VAL_SHIFTS]
    
    # Plot fluctuations:
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
    # Plot fit:
    plt.plot(
        atom_numbers[rel_fluct_target],
        [ftn_fctn(i, popt.exp.loc[rel_fluct_target], popt.offset.loc[rel_fluct_target]) for i in atom_numbers[rel_fluct_target]],
        color=plot_colors[idx],
        label='Fit: '+r'$\frac{\Delta N_0^2}{N} \propto N^{1 + \gamma}; \gamma_{\mathrm{fit}} = $'+f'{popt.exp.mean():.2}'+r'$\ ; \gamma_{\mathrm{theo}} = -\frac{2}{3}$'
        )            
ylabel = r'$\Delta N_{{0}}^2|_{{\frac{{U}}{{J}}={0}}}\ /\ N$'.format(UJ_SCALING)
fit_plot_atom_numbers = np.linspace(min([min(atom_numbers[i]) for i in REL_FLUCT_TARGETS]), max([max(atom_numbers[i]) for i in REL_FLUCT_TARGETS]), 100)
plt.xlabel(r'$N$')
plt.ylabel(ylabel)
plt.title(f'Scaling of the ground state occupation fluctuations with the atom number at U/J = {UJ_SCALING}')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

    
# scaling(CTRL_VAL_SHIFTS)

                 