#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:23:05 2022

@author: jp
"""

# Standard library imports:
import numpy as np
from matplotlib import pyplot as plt

# Local imports:
from setup import setup, load_data, get_atom_numbers_all_shots
from plot_colors import get_plot_colors
from post_selection import set_ctrl_vals_for_ps, post_select, momentum_select
                            

""" ---------- INPUT ---------- """
USE_ATOM_NUMBER_CALIB_UJ = True
CTR_VAL_SHIFT = None  # If not calib: int/float; else: None
SAVE_VARIANCE_FIG = False
REL_FLUCT_TARGETS = [0.42, 3, 6, 8]#5.7, 10.7, 15.7]#[1, 3, 5]
ETA = 0.53

#%% Calculate and plot variance:
    
def plot_variance(REL_FLUCT_TARGETS):
    
    # Prepare figure:
    fig, ax1 = plt.subplots(figsize=(19, 9))
    plot_colors = get_plot_colors(
        'qualitative',
        max([len(REL_FLUCT_TARGETS), 3]),
        name='Set1'
        )
    
    for idx, rel_fluct_target in enumerate(REL_FLUCT_TARGETS):
                
        # Post select for each target value of relative fluctuations:
        cutoff_for_rel_target, ps_atom_numbers, ps_distr = post_select(
            rel_fluct_target,
            uj_vals,
            atom_numbers_all_shots,
            recentered_data,
            ps_ctrl_vals
            )
        mom_ps_atom_numbers = momentum_select(
                                    uj_vals,
                                    ps_atom_numbers,
                                    ps_distr
                                    )
        # Calculate variance and normalize with detection efficiency:
        variance = mom_ps_atom_numbers.var()
        relative_fluctuations = variance.divide(5e3) / (ETA**2)
        relative_fluctuations_error = variance.mul(
                                    np.sqrt(
                                        2 / (mom_ps_atom_numbers.count() - 1)
                                        )).divide(5e3) / (ETA**2)
        
        
        # Calculate relative fluctuations of post-selected shots:
        fluct_std_perc = 100 * ps_atom_numbers.std().divide(
                                                        ps_atom_numbers.mean()
                                                        ).mean()

        # Predictions for shot-noise fluctuations:
        shot_noise_norm =  mom_ps_atom_numbers.mean() / ETA / 5e3
        
        # Predictions for shot-to-shot fluctuations:
        sts = (mom_ps_atom_numbers.mean().pow(2)
                * (fluct_std_perc / 100)**2 / (ETA**2) / 5e3)
        sts_error = (mom_ps_atom_numbers.std().pow(2)
                     * (fluct_std_perc / 100)**2 / (ETA**2) / 5e3)
        sts += shot_noise_norm
        sts_error += shot_noise_norm
        
        # Plot:
        plot_label_rf = (r'$\frac{\Delta N}{N} = $'
                         + f'{fluct_std_perc:.1f}%  - Ground State Fluct.')
        plot_label_sts = (r'$\frac{\Delta N}{N} = $'
                         + f'{fluct_std_perc:.1f}%  - Shot-to-shot + Shot-noise Fluct.')
        ylabel = r'$\Delta N_{0}^2\ /\ N$'
        ax1.errorbar(
            uj_vals,
            relative_fluctuations,
            yerr=relative_fluctuations_error,
            color=plot_colors[idx],
            marker='o',
            markersize=5,
            linewidth=0,
            elinewidth=1.5,
            capsize=4.0,
            label=plot_label_rf,
            zorder=1/(idx+1),
            )
        ax1.plot(
            uj_vals,
            sts,
            color=plot_colors[idx],
            linewidth=1,
            label=plot_label_sts
            )
        ax1.fill_between(
            uj_vals,
            sts-sts_error,
            sts+sts_error,
            alpha=0.3,
            facecolor=plot_colors[idx]
            )
    ax1.grid(visible=True)
    ax1.set_xlabel('Lattice depth [U/J]')
    ax1.set_ylabel(ylabel)
    plt.title('Relative atom number fluctuations in the BEC')        
    lines, labels = [ax.get_legend_handles_labels() for ax in fig.axes][0]
    fig.legend(
        lines,
        labels,
        ncol=2,
        framealpha=1,
        loc='upper right',
        bbox_to_anchor=(0.5, 0.1, 0.49, 0.86)
        )
    plt.tight_layout()
    plt.show()
  
plot_variance(REL_FLUCT_TARGETS)      
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
    plot_variance(REL_FLUCT_TARGETS)
