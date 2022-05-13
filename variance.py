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
from post_selection import set_ctrl_vals_for_ps, post_select, momentum_select
                            

""" ---------- INPUT ---------- """
USE_ATOM_NUMBER_CALIB_UJ = True
CTR_VAL_SHIFT = None  # If not calib: int/float; else: None
SAVE_VARIANCE_FIG = False
REL_FLUCT_TARGETS = [1, 3, 5]#[0.7, 5.7, 15.7]
ETA = 0.53

#%% Calculate and plot variance:
    
def plot_variance(REL_FLUCT_TARGETS):
    
    # Prepare figure:
    fig, ax1 = plt.subplots(figsize=(16, 9))
    plot_colors = get_plot_colors(
        'qualitative',
        len(REL_FLUCT_TARGETS),
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
                                        )).divide(5e3)
        
        # Calculate relative fluctuations of post-selected shots:
        fluct_std_perc = 100 * ps_atom_numbers.std().divide(
                                                        ps_atom_numbers.mean()
                                                        ).mean()
        
        # Predictions for other fluctuations:
        sts = (mom_ps_atom_numbers.mean().pow(2)
                * (fluct_std_perc / 100)**2 / (ETA**2) / 5e3)            
       
        shot_noise_norm =  mom_ps_atom_numbers.mean() / ETA / 5e3
        
        sts += shot_noise_norm
        
        # Plot:
        plot_label_rf = (r'$\frac{\Delta N}{N} = $'
                         + f'{fluct_std_perc:.2f}%  - Rel. Fluct.')
        plot_label_sts = (r'$\frac{\Delta N}{N} = $'
                         + f'{fluct_std_perc:.2f}%  - Shot-to-shot + Shot-noise Fluct.')
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
            linestyle='--',
            label=plot_label_sts
            )
    ax1.grid()
    ax1.set_xlabel('Lattice depth [U/J]')
    ax1.set_ylabel(ylabel)
    plt.title('Relative atom number fluctuations in the BEC')        
    lines, labels = [ax.get_legend_handles_labels() for ax in fig.axes][0]
    fig.legend(
        lines,
        labels,
        loc='upper right',
        bbox_to_anchor=(0.5, 0.1, 0.4, 0.8)
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

#%%




    
  
    
    
    
    
    
    
    # variance_plot = [variance[uj] for uj in sorted(uj_vals)]
    
    # bec_peak_atom_numbers_plot = [np.sqrt(np.mean(bec_atoms[uj])) for uj in sorted(uj_vals)]
    # atom_numbers_plot = [np.mean(shot_atoms[uj]) for uj in sorted(uj_vals)]
    # Plot fluctuations:
    # real_atom_nb_k = np.mean([np.mean(shot_atoms[uj]) for uj in sorted(uj_vals)])/1000
    # fluct_std_perc = rel_fluct(rel_fluct_target)#100*np.mean([np.std(list(shot_atom_numbers[rel_fluct_target][uj].values()))/np.mean(list(shot_atom_numbers[rel_fluct_target][uj].values())) for uj in sorted(uj_vals) for uj in sorted(uj_vals)])
    
    # sts = [fluct_std_perc/100 *   real_atom_nb_k * 1000 * condensed_fraction[idx]**2 for idx, _ in enumerate(uj_vals)]
    
    # plot_label_var = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}% fluct'
    # plot_label_bec = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}%  - ' + r'$\sqrt{\overline''{N_{\mathrm{BEC}}}}$'
    # plot_label_sts = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}%  - Shot-to-shot Fluct.'
    # plot_label_sf = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}% fluct'
    # plot_label_an = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}% fluct'
    # ax1.plot(sorted(uj_vals), variance_plotrelative_fluctuations, color=plot_colors, label=plot_label_var)#, marker='*')
    # ax2.plot(sorted(uj_vals), bec_peak_atom_numbers_plot, color='k', linestyle=':', linewidth=0.5, label=plot_label_bec)
    # ax2.scatter(sorted(uj_vals), atom_numbers_plot, label=plot_label_an)
    
    # ax1.plot(
    #     sorted(uj_vals),
    #     sts,
    #     color=plot_colors[0],
    #     label=plot_label_sts,
    #     linestyle='--'
    #     )
    
    
    # ax2.plot(
    #     sorted(uj_vals),
    #     [calibrate_lattice_atom_number(uj)*n_factor for uj in sorted(uj_vals)],
    #     color=plot_colors,
    #     linestyle=':',
    #     linewidth=0.75,
    #     label='Fit from calibration'
    #     )
    # ax1.plot(
    #     sorted(uj_vals),
    #     shot_noise_norm,
    #     label='Shot Noise',
    #     color='k',
    #     linestyle='-.'
    #     )
    # central_density_norm_an = [i * np.mean(bec_peak_atom_numbers_plot) for i in central_density_norm]
    # ax2.plot(sorted(uj_vals), central_density_norm_an, color='k', linestyle='-.', linewidth=1.5, label=r'$\rho_0$ [a.u.]')
    # ax1.plot(sorted(uj_vals), politzer_norm_fluct, color='grey', linestyle='--', linewidth=1.5, label=r'$\Delta N_0^2 \propto \frac{\zeta(2)}{\zeta(3)}\ N\ \left(\frac{T}{T_c^0}\right)^3$ (Politzer)')
    
    # ax1.set_ylabel(r'$(\Delta N_{BEC})^2$')
    # ax2.set_ylabel(r'$\sqrt{\overline{\mathrm{Atom Number}}}$')
    # plt.title('variance of the atom number in the BEC for ' + r'$k_{max}$' + ' = ' + f"{k_max} " + r'$k_d$')
    
    