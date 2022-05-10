#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:54:34 2022

@author: jp

Post-selection of experimental shots for a given target of relative
fluctuations of the atom number.

Specify the dataset in terms of its U/J value and the relative fluctuation
target (in %). The central number can be set by the calibration of the atom
numbers as a function of U/J for 5k atoms in the trap
(USE_ATOM_NUMBER_CALIB_UJ); otherwise the mean atom number will be used for
each dataset. The function returns the indices of the post-selected shots.
The result of the post-selection can be plotted (PLOT_POST_SELECTION) and
saved (SAVE_POST_SELECTION_FIG).
"""

# Standard library imports:
import numpy as np
from matplotlib import pyplot as plt

# Local imports:
from setup import setup, load_data, get_atom_numbers_all_shots
from helper_functions import poly_n
from rel_fluct_cutoff_fctn import cutoff_from_rel_fluct
from plot_colors import get_plot_colors

""" ---------- INPUT ---------- """
REL_FLUCT_TARGET = 0.7  # %
DATASET = 5  # U/J
USE_ATOM_NUMBER_CALIB_UJ = True
CTR_VAL_SHIFT = None  # If not calib: int/float; else: None
PLOT_POST_SELECTION = True
SAVE_POST_SELECTION_FIG = False

#%% Load atom number calibration for U/J:
    
def set_ctrl_vals_for_ps(
        USE_ATOM_NUMBER_CALIB_UJ,
        CTR_VAL,
        lattice_atom_number_calibration,
        uj_vals
        ):
    
    if USE_ATOM_NUMBER_CALIB_UJ:
    
        ps_ctrl_vals = {
            uj: poly_n(uj, *list(reversed(
                lattice_atom_number_calibration.values()
                )))
            for uj in uj_vals
            }
        
    else:
        
        ps_ctrl_vals = {
            uj: poly_n(uj, *list(reversed(
                lattice_atom_number_calibration.values()
                )))
            for uj in uj_vals
            }
    
    return ps_ctrl_vals
    
#%% Post-Selection:
    
def post_select(REL_FLUCT_TARGET, atom_numbers_all_shots, ps_ctrl_vals):
        
    cutoff_for_rel_target = cutoff_from_rel_fluct(REL_FLUCT_TARGET)

    ps_indices = dict.fromkeys(sorted(uj_vals))

    for uj in sorted(uj_vals):

        calib_mean = ps_ctrl_vals[uj]
        ps_indices[uj] = np.where(
            abs(
                atom_numbers_all_shots[uj] - calib_mean
                ) / atom_numbers_all_shots[uj] < (cutoff_for_rel_target / 100)
            )[0]
        
    post_selection = {
        uj: atom_numbers_all_shots[uj][ps_indices[uj]]
        for uj in sorted(uj_vals)
        }
    eff_std = {
        uj: post_selection[uj].std()
        for uj in post_selection
        }
    eff_rel_fluct_perc = {
        uj: 100 * post_selection[uj].std() / post_selection[uj].mean()
        for uj in post_selection
        }

    return cutoff_for_rel_target, post_selection, eff_std, eff_rel_fluct_perc

#%% Plot Post-Selection result:
    
def plot_post_selection_result(
        DATASET,
        cutoff_for_rel_target,
        post_selection,
        eff_std,
        eff_rel_fluct_perc,
        ps_ctrl_vals
        ):
    
    plot_colors = get_plot_colors('qualitative', 3, name='Set1')
    
    label_post_selection = (r'PS: $\Delta N^{\mathrm{MCP}} = $'
                            +f'{eff_std[DATASET]:.0f}, '
                            +r'$\frac{\Delta N}{N} = $'
                            +f'{eff_rel_fluct_perc[DATASET]:.1f}%')
    title = (f'U/J = {DATASET}\n'
             +r'$F_{\mathrm{PS}} = $'
             +f'{cutoff_for_rel_target:.0f}% '
             +r'$\rightarrow$'
             +f' {post_selection[DATASET].count()} Post-Selected Shots'
             )

    plt.figure(figsize=(8, 4))
    plt.scatter(
        atom_numbers_all_shots.index,
        atom_numbers_all_shots[DATASET].values,
        color=plot_colors[1],
        s=10,
        alpha=0.75,
        zorder=1,
        label='All shots'
        )    
    plt.scatter(
        post_selection[DATASET].index,
        post_selection[DATASET].values,
        color=plot_colors[0],
        marker='o',
        s=25,
        edgecolors='k',
        linewidths=0.1,
        alpha=0.75,
        zorder=2,
        label=label_post_selection
        )
    plt.axhline(
        ps_ctrl_vals[DATASET],
        color=plot_colors[2],
        linestyle='--',
        linewidth=1.5,
        zorder=0,
        label=r'$N_{\mathrm{Calib}}^{\mathrm{MCP}}$'
        )
    plt.xlabel('Shots')
    plt.ylabel(r'Detected atom number $N^{\mathrm{MCP}}$')    
    plt.title(title) 
    plt.legend(loc='upper right', framealpha=1)
    plt.grid()
    plt.tight_layout()    
    plt.show()

#%% Execution:

if __name__ == '__main__':

    data_basepath, figure_savepath, lattice_atom_number_calibration = setup()
    recentered_data, uj_vals = load_data(data_basepath)
    atom_numbers_all_shots = get_atom_numbers_all_shots(
                                recentered_data,
                                uj_vals
                                )
    ps_ctrl_vals = set_ctrl_vals_for_ps(
                    USE_ATOM_NUMBER_CALIB_UJ,
                    CTR_VAL_SHIFT,
                    lattice_atom_number_calibration,
                    uj_vals
                    )    
    (
     cutoff_for_rel_target,
     post_selection,
     eff_std,
     eff_rel_fluct_perc
     ) = post_select(REL_FLUCT_TARGET, atom_numbers_all_shots, ps_ctrl_vals)
    plot_post_selection_result(
        DATASET,
        cutoff_for_rel_target,
        post_selection,
        eff_std,
        eff_rel_fluct_perc,
        ps_ctrl_vals
        )
    