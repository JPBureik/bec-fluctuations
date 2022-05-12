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
from helper_functions import poly_n
from rel_fluct_cutoff_fctn import cutoff_from_rel_fluct
from plot_colors import get_plot_colors
from post_selection import (set_ctrl_vals_for_ps, post_select,
                            plot_post_selection_result)

""" ---------- INPUT ---------- """
USE_ATOM_NUMBER_CALIB_UJ = True
CTR_VAL_SHIFT = None  # If not calib: int/float; else: None
PLOT_POST_SELECTION = True
SAVE_POST_SELECTION_FIG = False

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
#%%
fig, ax1 = plt.subplots(figsize=(16, 9))
i = 0
def plot_variance(REL_FLUCT_TARGET, i):

# REL_FLUCT_TARGET = 0.7  # %

    (
     cutoff_for_rel_target,
     ps_atom_numbers,
     ps_momemtum_distr
     ) = post_select(
        REL_FLUCT_TARGET,
        uj_vals,
        atom_numbers_all_shots,
        recentered_data,
        ps_ctrl_vals
        )
    
    # Calculate variance:
    
    k_min = 0
    k_max = 0.15
    
    # Count number of atoms in BEC and in entire shot:
    bec_peak_atom_numbers = dict.fromkeys(uj_vals)
    relative_fluctuations = dict.fromkeys(uj_vals)
    relative_fluctuations_error = dict.fromkeys(uj_vals)
    shot_atom_numbers = dict.fromkeys(uj_vals)
    variance = dict.fromkeys(uj_vals)
    bec_peak_atom_numbers = dict.fromkeys(uj_vals)
    for uj in uj_vals:
        bec_peak_atom_numbers[uj] = dict.fromkeys(ps_momemtum_distr[uj].index.dropna())
        relative_fluctuations[uj] = dict.fromkeys(ps_momemtum_distr[uj].index.dropna())
        relative_fluctuations_error[uj] = dict.fromkeys(ps_momemtum_distr[uj].index.dropna())
        shot_atom_numbers[uj] = dict.fromkeys(ps_momemtum_distr[uj].index.dropna())
        for run in ps_momemtum_distr[uj]['k_h'].index.dropna():
    
            indices = dict.fromkeys(['k_h', 'k_m45', 'k_p45'])
    
            for axis in indices:
                indices[axis] = (k_min < abs(ps_momemtum_distr[uj][axis][run])) * (abs(ps_momemtum_distr[uj][axis][run]) < k_max)
            indices_all = np.where(indices['k_h'] * indices['k_m45'] * indices['k_p45'])
            assert len(list(ps_momemtum_distr[uj]['k_h'][run][indices_all])) == len(list(ps_momemtum_distr[uj]['k_m45'][run][indices_all])) == len(list(ps_momemtum_distr[uj]['k_p45'][run][indices_all]))
            bec_peak_atom_numbers[uj][run] = len(list(ps_momemtum_distr[uj]['k_h'][run][indices_all]))
            assert len(list(ps_momemtum_distr[uj]['k_h'][run])) == len(list(ps_momemtum_distr[uj]['k_m45'][run])) == len(list(ps_momemtum_distr[uj]['k_p45'][run]))
            shot_atom_numbers[uj][run] = len(list(ps_momemtum_distr[uj]['k_h'][run]))
    
    
    # Calculate variance and normalize:
    for uj in uj_vals:
        variance[uj] = np.var(list(bec_peak_atom_numbers[uj].values()))
        relative_fluctuations[uj] = variance[uj]/np.mean(list(shot_atom_numbers[uj].values()))
        relative_fluctuations_error[uj] = variance[uj]*np.sqrt(2/(len(list(bec_peak_atom_numbers[uj].values()))-1))/np.mean(list(shot_atom_numbers[uj].values()))
    
    # Plot variance
    
    
    
    
    # ax2 = ax1.twinx()
    
    plot_color_str = """228,26,28
    55,126,184
    77,175,74
    152,78,163
    255,127,0
    255,255,51"""
    
    def make_plot_colors(color_str):
        return [tuple([int(i)/255 for i in color_str.split('\n')[j].split(',')])
                for j in range(len(color_str.split('\n')))]
    
    plot_colors = make_plot_colors(plot_color_str)
    
    def condensed_fraction_scaling(uj):
    
        if uj > 25:
            return 0
        else:
            return (1 - uj/26)**0.6
    
    condensed_fraction = [condensed_fraction_scaling(uj) for uj in sorted(uj_vals)]
    
    shot_noise = [condensed_fraction[idx] * 0.53 * 5e3 for idx, _ in enumerate(uj_vals)]
    shot_noise_norm = [shot_noise[idx] / 5e3 for idx, _ in enumerate(shot_noise)]
    
    
    
    
    
    
    variance_plot = [variance[uj] for uj in sorted(uj_vals)]
    relative_fluctuations_plot = [relative_fluctuations[uj] for uj in sorted(uj_vals)]
    relative_fluctuations_error_plot = [relative_fluctuations_error[uj] for uj in sorted(uj_vals)]
    bec_peak_atom_numbers_plot = [np.sqrt(np.mean(list(bec_peak_atom_numbers[uj].values()))) for uj in sorted(uj_vals)]
    atom_numbers_plot = [np.mean(list(shot_atom_numbers[uj].values())) for uj in sorted(uj_vals)]
    # Plot fluctuations:
    real_atom_nb_k = np.mean(np.mean([list(shot_atom_numbers[uj].values()) for uj in sorted(uj_vals)]))/1000
    # fluct_std_perc = rel_fluct(rel_fluct_target)#100*np.mean([np.std(list(shot_atom_numbers[rel_fluct_target][uj].values()))/np.mean(list(shot_atom_numbers[rel_fluct_target][uj].values())) for uj in sorted(uj_vals) for uj in sorted(uj_vals)])
    fluct_std_perc = 100*np.mean([np.std(list(shot_atom_numbers[uj].values()))/np.mean(list(shot_atom_numbers[uj].values())) for uj in sorted(uj_vals) for uj in sorted(uj_vals)])
    
    sts = [fluct_std_perc/100 *   real_atom_nb_k * 1000 * condensed_fraction[idx]**2 for idx, _ in enumerate(uj_vals)]
    
    plot_label_var = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}% fluct'
    plot_label_bec = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}%  - ' + r'$\sqrt{\overline''{N_{\mathrm{BEC}}}}$'
    plot_label_rf = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}%  - Rel. Fluct.'
    plot_label_sts = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}%  - Shot-to-shot Fluct.'
    plot_label_sf = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}% fluct'
    plot_label_an = r'$\frac{\Delta N}{N} = $' + f'{fluct_std_perc:.2f}% fluct'
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
    
    ax1.errorbar(
        sorted(uj_vals),
        relative_fluctuations_plot,
        yerr=relative_fluctuations_error_plot,
        color=plot_colors[i],
        marker='o',
        markersize=5,
        linewidth=0,
        elinewidth=1.5,
        capsize=4.0,
        label=plot_label_rf,
        zorder=i
        )
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
    ax1.grid()
    ax1.set_xlabel('Lattice depth [U/J]')
    ax1.set_ylabel(r'$(\Delta N_{0}^{\mathrm{MCP}})^2\ /\ N^{\mathrm{MCP}}$')
    # ax1.set_ylabel(r'$(\Delta N_{BEC})^2$')
    # ax2.set_ylabel(r'$\sqrt{\overline{\mathrm{Atom Number}}}$')
    plt.title('Relative atom number fluctuations in the BEC center for ' + r'$k_{max}$' + ' = ' + f"{k_max} " + r'$k_d$')
    # plt.title('variance of the atom number in the BEC for ' + r'$k_{max}$' + ' = ' + f"{k_max} " + r'$k_d$')
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    
    return lines, labels

for REL_FLUCT_TARGET in reversed([15.7, 5.7, 0.7]):
    lines, labels = plot_variance(REL_FLUCT_TARGET, i)
    i+=1
fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.5, 0.1, 0.4, 0.8))
plt.tight_layout()
plt.show()