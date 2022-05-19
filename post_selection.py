#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:54:34 2022

@author: jp

Post-selection of experimental shots in terms of atom number and momentum.

- post_select(): 
    Post-selection of experimental data in terms of atom number.
    Specify the dataset in terms of its U/J value and the relative fluctuation
    target (in %). The central number can be set by the calibration of the atom
    numbers as a function of U/J for 5k atoms in the trap
    (USE_ATOM_NUMBER_CALIB_UJ); otherwise the mean atom number will be used for
    each dataset. The function returns the post-selected shots. The result of
    the post-selection can be plotted (PLOT_POST_SELECTION) and saved
    (SAVE_POST_SELECTION_FIG).

- momentum_select():
    Post-selection of experimental data in terms of momentum.
    Retains only those atoms in each shot that have k_min < |k| < k_max for
    all three directions simultaneously. Default values: k_min = 0 and
    k_max = 0.15 in units of k_d.
"""

# Standard library imports:
import numpy as np
import pandas as pd
import pickle as pl
from matplotlib import pyplot as plt

# Local imports:
from setup import setup, load_data, get_atom_numbers_all_shots
from helper_functions import poly_n
from rel_fluct_cutoff_fctn import cutoff_from_rel_fluct
from plot_colors import get_plot_colors

""" ---------- INPUT ---------- """
REL_FLUCT_TARGET = 15.7  # %
DATASET = 5  # U/J
USE_ATOM_NUMBER_CALIB_UJ = True
CTR_VAL_FACTOR = None  # If not calib: int/float; else: None
PLOT_POST_SELECTION = True
SAVE_POST_SELECTION_FIG = False

#%% Load atom number calibration for U/J:
    
def set_ctrl_vals_for_ps(
        USE_ATOM_NUMBER_CALIB_UJ,
        CTR_VAL_FACTOR,
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
                ))) * CTR_VAL_FACTOR
            for uj in uj_vals
            }
    
    return ps_ctrl_vals
    
#%% Post-Selection:
    
def post_select(
        REL_FLUCT_TARGET,
        uj_vals,
        atom_numbers_all_shots,
        recentered_data,
        ps_ctrl_vals
        ):

    """Post-selection of experimental data in terms of atom number."""
    
    # Get fluctuation cutoff that corresponds to target:
    cutoff_for_rel_target = cutoff_from_rel_fluct(REL_FLUCT_TARGET)

    # Get indices for shots outside the fluctuation target range:
    ps_indices = pd.Series(data={
        uj: np.where(~(
            abs(atom_numbers_all_shots[uj] - ps_ctrl_vals[uj])
            / atom_numbers_all_shots[uj]
            < (cutoff_for_rel_target / 100)
            ))[0]
        for uj in sorted(uj_vals)
        })
        
    # Apply index mask to atom numbers and momentum distribution:
    ps_atom_numbers = pd.DataFrame.from_records(atom_numbers_all_shots)
    ps_distr = dict.fromkeys(uj_vals)
    for uj in uj_vals:
        ps_atom_numbers[uj].loc[ps_indices[uj]] = np.nan
        ps_distr[uj] = pd.DataFrame.from_records(recentered_data[uj])
        # Discard entries for indexes not current in dataset:
        ps_indices_in_distr = ps_indices[uj][ps_indices[uj]<len(ps_distr[uj])]
        ps_distr[uj].loc[ps_indices_in_distr] = np.nan
    
    return cutoff_for_rel_target, ps_atom_numbers, ps_distr

#%% Momentum Selection:

def momentum_select(
        uj_vals,
        ps_atom_numbers,
        ps_distr,
        k_min=0,
        k_max=0.15
        ):

    """Post-selection of experimental data in terms of momentum."""

    # Get atom numbers for shots within the momentum target range:
    mom_ps_atom_numbers = pd.DataFrame.from_records(ps_atom_numbers)
    for uj in uj_vals:
        for run in ps_atom_numbers[uj][ps_atom_numbers[uj].notnull()].index:
            # Get indices where k_min < |k| < k_max for all axes individually:
            mom_ps_indices_single_axis = {
                axis: ((k_min < abs(ps_distr[uj][axis][run]))
                       * (abs(ps_distr[uj][axis][run]) < k_max))
                for axis in ['k_h', 'k_m45', 'k_p45']
                }
            # Get intersection of indices for all axes simultaneously:
            mom_ps_atom_numbers[uj].at[run] = np.count_nonzero(
                                        mom_ps_indices_single_axis['k_h']
                                        * mom_ps_indices_single_axis['k_m45']
                                        * mom_ps_indices_single_axis['k_p45']
                                        )  
            
    return mom_ps_atom_numbers

#%% Plot Post-Selection result:
    
def plot_post_selection_result(
        DATASET,
        cutoff_for_rel_target,
        atom_numbers_all_shots,
        ps_atom_numbers,
        ps_ctrl_vals
        ):
    
    plot_colors = get_plot_colors('qualitative', 3, name='Set1')
    
    ps_atom_numbers_std = {
        uj: ps_atom_numbers[uj].std()
        for uj in ps_atom_numbers
        }
    ps_atom_numbers_rel_fluct_perc = {
        uj: 100 * ps_atom_numbers[uj].std() / ps_atom_numbers[uj].mean()
        for uj in ps_atom_numbers
        }
    
    label_post_selection = (r'PS: $\Delta N^{\mathrm{MCP}} = $'
                            +f'{ps_atom_numbers_std[DATASET]:.0f}, '
                            +r'$\frac{\Delta N}{N} = $'
                            +f'{ps_atom_numbers_rel_fluct_perc[DATASET]:.1f}%')
    title = (f'U/J = {DATASET}\n'
             +r'$F_{\mathrm{PS}} = $'
             +f'{cutoff_for_rel_target:.0f}% '
             +r'$\rightarrow$'
             +f' {ps_atom_numbers[DATASET].count()} Post-Selected Shots'
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
        ps_atom_numbers[DATASET].index,
        ps_atom_numbers[DATASET].values,
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
     ps_atom_numbers,
     ps_distr
     ) = post_select(
         REL_FLUCT_TARGET,
         uj_vals,
         atom_numbers_all_shots,
         recentered_data,
         ps_ctrl_vals
         )
    plot_post_selection_result(
        DATASET,
        cutoff_for_rel_target,
        atom_numbers_all_shots,
        ps_atom_numbers,
        ps_ctrl_vals
        )
    