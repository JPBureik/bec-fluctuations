#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:54:34 2022

@author: jp

Post-selection of experimental shots for a given target of relative
fluctuations of the atom number.

Specify the dataset in terms of its U/J value and the relative fluctuation
target. The central number can be set by the calibration of the atom numbers
as a function of U/J for 5k atoms in the trap (USE_ATOM_NUMBER_CALIB_UJ);
otherwise the mean atom number will be used for each dataset. The result of
the post-selection can be plotted (PLOT_POST_SELECTION) and saved
(SAVE_POST_SELECTION_FIGURE).
"""

# Standard library imports:
import numpy as np
from matplotlib import pyplot as plt

# Local imports:
from setup import setup, load_data, get_atom_numbers_all_shots
from helper_functions import poly_n
from rel_fluct_cutoff_fctn import cutoff_from_rel_fluct
from plot_colors import get_plot_colors

#%% Execution:

if __name__ == '__main__':

    data_basepath, figure_savepath, lattice_atom_number_calibration = setup()
    recentered_data, uj_vals = load_data(data_basepath)
    atom_numbers_all_shots = get_atom_numbers_all_shots(
                                recentered_data,
                                uj_vals
                                )
    
#%%

plot_colors = get_plot_colors('qualitative', 3, name='Set1')

calibrated_atom_numbers = {
    uj: poly_n(uj, *list(reversed(lattice_atom_number_calibration.values())))
    for uj in uj_vals
    }

rel_fluct_target = 0.7
cutoff_for_rel_target = cutoff_from_rel_fluct(rel_fluct_target)


def post_select(rel_fluct_target, ctr_val=calibrated_atom_numbers):
    
    cutoff_for_rel_target = cutoff_from_rel_fluct(rel_fluct_target)

    ps_indices = dict.fromkeys(sorted(uj_vals))

    for uj in sorted(uj_vals):

        calib_mean = ctr_val[uj]
        ps_indices[uj] = np.where(abs(atom_numbers_all_shots[uj] - calib_mean) / atom_numbers_all_shots[uj] < (cutoff_for_rel_target/100))[0]

    return ps_indices

post_selection = {uj: atom_numbers_all_shots[uj][post_select(rel_fluct_target)[uj]] for uj in sorted(uj_vals)}
eff_std = {uj: post_selection[uj].std() for uj in post_selection}
eff_rel_fluct_perc = {uj: 100 * post_selection[uj].std() / post_selection[uj].mean() for uj in post_selection}

def plot_single(uj):
    plt.figure(figsize=(8, 4))
    plt.scatter(
        atom_numbers_all_shots.index,
        atom_numbers_all_shots[uj].values,
        color=plot_colors[1],
        label='All shots',
        s=10,
        alpha=0.75,
        zorder=1
        )    
    plt.scatter(
        post_selection[uj].index,
        post_selection[uj].values,
        color=plot_colors[0],
        marker='o',
        s=25,
        edgecolors='k',
        linewidths=0.1,
        alpha=0.75,
        zorder=2,
        label=r'PS: $\Delta N^{\mathrm{MCP}} = $'+f'{eff_std[uj]:.0f}, '+r'$\frac{\Delta N}{N} = $'+f'{eff_rel_fluct_perc[uj]:.1f}%'
        )
    plt.axhline(
        calibrated_atom_numbers[uj],
        color=plot_colors[2],
        linestyle='--',
        linewidth=1.5,
        zorder=0,
        label=r'$N_{\mathrm{Calib}}^{\mathrm{MCP}}$'# = $'+f'{int(calibrated_atom_numbers[uj_vals[idx]])}'
        )
    plt.xlabel('Shots')
    plt.ylabel(r'Detected atom number $N^{\mathrm{MCP}}$')    
    plt.title(f'U/J = {uj}\n'+r'$F_{\mathrm{PS}} = $'+f'{cutoff_for_rel_target:.0f}% '+r'$\rightarrow$'+f' {len([idx for idx, _ in enumerate(post_selection[uj].index)])} Post-Selected Shots')#, '+r'$\bar{N}_{\mathrm{PS}} = $'+f'{np.mean(post_selection[uj].values):.0f}')    
    plt.legend(loc='upper right', framealpha=1)
    plt.grid()
    plt.tight_layout()    
    plt.show()
    