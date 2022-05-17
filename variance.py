#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:23:05 2022

@author: jp
"""

# Standard library imports:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Local imports:
from setup import setup, load_data, get_atom_numbers_all_shots
from plot_colors import get_plot_colors
from post_selection import (set_ctrl_vals_for_ps, post_select,
                            momentum_select, plot_post_selection_result)
                            

""" ---------- INPUT ---------- """
USE_ATOM_NUMBER_CALIB_UJ = True
CTR_VAL_SHIFT = None  # If not calib: int/float; else: None
PLOT_VARIANCE = True
SAVE_VARIANCE_FIG = False
REL_FLUCT_TARGETS = [0.7, 6.1, 8, 15]
ETA = 0.53

#%% Calculate and plot variance:
    
def variance_norm(
        uj_vals,
        atom_numbers_all_shots,
        recentered_data,
        ps_ctrl_vals,
        REL_FLUCT_TARGETS,
        plot_ps=False
        ):
    
    """Calculate the variance of the aton number in the measurement volume and
    normalize by the total atom number in terms of in-trap quantities.
    Predicitions for shot noise and shot to shot fluctuations extracted from
    experimental data excluding the rise of fluctuations at the phase
    transition."""
    
    # Prepare data containers:
    variance = pd.DataFrame(
                    data=None,
                    index=uj_vals,
                    columns=REL_FLUCT_TARGETS
                    )
    relative_fluctuations = pd.DataFrame(
                    data=None,
                    index=uj_vals,
                    columns=REL_FLUCT_TARGETS
                    )
    relative_fluctuations_error = pd.DataFrame(
                    data=None,
                    index=uj_vals,
                    columns=REL_FLUCT_TARGETS
                    )
    fluct_std_perc = pd.Series(
                    data=None,
                    index=REL_FLUCT_TARGETS,
                    dtype=float
                    )
    shot_noise_norm = pd.DataFrame(
                    data=None,
                    index=uj_vals,
                    columns=REL_FLUCT_TARGETS
                    )
    sts = pd.DataFrame(
                    data=None,
                    index=uj_vals,
                    columns=REL_FLUCT_TARGETS
                    )
    sts_error = pd.DataFrame(
                    data=None,
                    index=uj_vals,
                    columns=REL_FLUCT_TARGETS
                    )
    ps_atom_numbers = dict.fromkeys(REL_FLUCT_TARGETS)
    
    # Post select for each target value of relative fluctuations:
    for rel_fluct_target in REL_FLUCT_TARGETS:
        
        (
            cutoff_for_rel_target,
            ps_atom_numbers[rel_fluct_target],
            ps_distr
            ) = post_select(
                rel_fluct_target,
                uj_vals,
                atom_numbers_all_shots,
                recentered_data,
                ps_ctrl_vals
                )
        if plot_ps:
            for uj in uj_vals:
                plot_post_selection_result(
                    uj,
                    cutoff_for_rel_target,
                    atom_numbers_all_shots,
                    ps_atom_numbers[rel_fluct_target],
                    ps_ctrl_vals
                    )
        mom_ps_atom_numbers = momentum_select(
                                    uj_vals,
                                    ps_atom_numbers[rel_fluct_target],
                                    ps_distr
                                    )
        # Calculate variance and normalize with detection efficiency:
        variance[rel_fluct_target] = mom_ps_atom_numbers.var()
        relative_fluctuations[rel_fluct_target] = (variance[rel_fluct_target]
                                                   .divide(mom_ps_atom_numbers[rel_fluct_target].mean().pow(2)/(ETA**2))
                                                   / (ETA**2))
        relative_fluctuations_error[rel_fluct_target] = (
                                        variance[rel_fluct_target].mul(
                                            np.sqrt(
                                        2 / (mom_ps_atom_numbers.count() - 1)
                                        )
                                        ).divide(mom_ps_atom_numbers[rel_fluct_target].mean().pow(2)/(ETA**2)) / (ETA**2)
                                        )
        
        
        # Calculate relative fluctuations of post-selected shots:
        fluct_std_perc[rel_fluct_target] = 100 * ps_atom_numbers[rel_fluct_target].std().divide(
                                                        mom_ps_atom_numbers[rel_fluct_target].mean()
                                                        ).mean()
    
        # Predictions for shot-noise fluctuations:
        shot_noise_norm[rel_fluct_target] =  (mom_ps_atom_numbers.mean().divide(mom_ps_atom_numbers[rel_fluct_target].mean().pow(2)/(ETA**2))
                                              / ETA)
        
        # Predictions for shot-to-shot fluctuations:
        sts[rel_fluct_target] = (mom_ps_atom_numbers.mean().pow(2).divide(mom_ps_atom_numbers[rel_fluct_target].mean().pow(2)/(ETA**2))
                * (fluct_std_perc[rel_fluct_target]
                   / 100)**2 / (ETA**2))
        sts_error[rel_fluct_target] = (mom_ps_atom_numbers.std().pow(2).divide(mom_ps_atom_numbers[rel_fluct_target].mean().pow(2)/(ETA**2))
                     * (fluct_std_perc[rel_fluct_target] / 100)**2
                     / (ETA**2))
        sts[rel_fluct_target] += shot_noise_norm[rel_fluct_target]
        sts_error[rel_fluct_target] += shot_noise_norm[rel_fluct_target]
        
    return (ps_atom_numbers, fluct_std_perc, relative_fluctuations,
            relative_fluctuations_error, sts, sts_error)
    
def plot_variance(
        uj_vals,
        REL_FLUCT_TARGETS,
        fluct_std_perc,
        relative_fluctuations,
        relative_fluctuations_error,
        sts,
        sts_error,
        title_add=''
        ):
    
    """Plot normalized variance of the ground state occupation and
    contributions of shot-to-shot and shot-noise fluctuations."""
    
    if PLOT_VARIANCE:
    
        # Prepare figure:
        fig, ax1 = plt.subplots(figsize=(13, 5))#figsize=(19, 9))
        plot_colors = get_plot_colors(
            'qualitative',
            max([len(REL_FLUCT_TARGETS), 3]),
            name='Set1'
            )
        
        for idx, rel_fluct_target in enumerate(REL_FLUCT_TARGETS):
                    
            # Plot:
            plot_label_rf = (r'$\frac{\Delta N}{N} = $'
                             + f'{fluct_std_perc[rel_fluct_target]:.1f}%  - '
                             + 'Ground State Fluct.')
            plot_label_sts = (r'$\frac{\Delta N}{N} = $'
                             + f'{fluct_std_perc[rel_fluct_target]:.1f}%  - '
                             + 'Shot-to-shot + Shot-noise Fluct.')
            ylabel = r'$\Delta N_{0}^2\ /\ N_0^2$'
            ax1.errorbar(
                uj_vals,
                relative_fluctuations[rel_fluct_target],
                yerr=relative_fluctuations_error[rel_fluct_target],
                color=plot_colors[idx],
                marker='o',
                markersize=10,
                linewidth=0,
                elinewidth=2.5,
                capsize=4.0,
                label=plot_label_rf,
                zorder=1/(idx+1),
                )
            ax1.plot(
                uj_vals,
                sts[rel_fluct_target],
                color=plot_colors[idx],
                linewidth=1,
                label=plot_label_sts
                )
            ax1.fill_between(
                uj_vals,
                sts[rel_fluct_target]-sts_error[rel_fluct_target],
                sts[rel_fluct_target]+sts_error[rel_fluct_target],
                alpha=0.3,
                facecolor=plot_colors[idx]
                )
        ax1.grid(visible=True, lw=0.25)
        ax1.set_xlabel('Lattice depth [U/J]')
        ax1.set_ylabel(ylabel)
        if title_add:
            title = title_add + ' - Relative atom number fluctuations in the BEC'
        else:
            title = 'Relative atom number fluctuations in the BEC'
        plt.title(title)        
        lines, labels = [ax.get_legend_handles_labels() for ax in fig.axes][0]
        fig.legend(
            lines,
            labels,
            ncol=1,
            framealpha=1,
            loc='upper right',
            # loc='upper left',
            # bbox_to_anchor=(0.5, 0.1, 0.49, 0.86)
            bbox_to_anchor=(0.5, 0.1, 0.49, 0.83)
            # bbox_to_anchor=(0.12, 0.1, -0.99, 0.76)
            )
        plt.tight_layout()
        plt.show()
    
#%% Execution:

if __name__ == '__main__':
    
    # data_basepath, figure_savepath, lattice_atom_number_calibration = setup()
    # recentered_data, uj_vals = load_data(data_basepath)
    # atom_numbers_all_shots = get_atom_numbers_all_shots(
    #                             recentered_data,
    #                             uj_vals
    #                             )
    # ps_ctrl_vals = set_ctrl_vals_for_ps(
    #                 USE_ATOM_NUMBER_CALIB_UJ,
    #                 CTR_VAL_SHIFT,
    #                 lattice_atom_number_calibration,
    #                 uj_vals
                    )
    (
     ps_atom_numbers,
     fluct_std_perc,
     relative_fluctuations,
     relative_fluctuations_error,
     sts,
     sts_error
     ) = variance_norm(
         uj_vals,
         atom_numbers_all_shots,
         recentered_data,
         ps_ctrl_vals,
         REL_FLUCT_TARGETS
         )
    plot_variance(
        uj_vals,
        REL_FLUCT_TARGETS,
        fluct_std_perc,
        relative_fluctuations,
        relative_fluctuations_error,
        sts,
        sts_error
        )
