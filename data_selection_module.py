#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:28:42 2022

@author: jp
"""

# Standard library imports:
import socket
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import boost_histogram as bh
from matplotlib import pyplot as plt
import pickle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math

# Local imports:
from mcpmeas.load_recentered_data import load_recentered_data

# Setup: Set file path to data directory:

hostname = socket.gethostname()

if hostname == 'catarella':
    data_basepath = '/home/jp/Documents/data/recentered_data/indiv_all'
    # Load atom number calibration:
    with open('/home/jp/Documents/prog/work/bec-fluctuations/lattice_atom_number_calibration_data.pickle', 'rb') as infile:
        lattice_atom_number_calibration = pickle.load(infile)

elif hostname == 'jp-MS-7C02':
    data_basepath = '/home/jp/Documents/data/recentered_data/indiv_all'
    with open('/home/jp/Documents/prog/work/bec-fluctuations/lattice_atom_number_calibration_data.pickle', 'rb') as infile:
        lattice_atom_number_calibration = pickle.load(infile)

# Load recentered data:
ujs = sorted([uj for uj in os.walk(data_basepath)][0][2])

datapaths = sorted([os.path.join(data_basepath, uj) for uj in ujs])

uj_vals = sorted([float(uj.split('uj')[1].split('_all')[0].replace('p','.')) for uj in ujs])

recentered_data = dict.fromkeys(uj_vals)

for idx, datapath in tqdm(enumerate(datapaths), desc='Loading data'):
    recentered_data[uj_vals[idx]] = load_recentered_data(datapath)

# Load fluct_calib:
with open('rel_fluct_calib.pickle', 'rb') as infile:
    rel_fluct_calib_params = pickle.load(infile)
    def poly_two(x, a, b, c): return a + x * b + x**2 * c
    def rel_fluct(fluct_cutoff): return poly_two(fluct_cutoff, *rel_fluct_calib_params)

#%% SET PARAMETERS:

# fluct_perc_vals = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 7.5, 10,]# 30]
# fluct_perc_vals = [0.1, 1, 5, 10]
fluct_perc_vals = np.linspace(0.1, 10, int(1e2))

retained_shots_numbers = dict.fromkeys(fluct_perc_vals)
retained_shots = dict.fromkeys(fluct_perc_vals)

for FLUCT_PERC in tqdm(fluct_perc_vals, desc='Counting atoms'):

    # FLUCT_PERC = 30
    SELECT_FOR_CALIB = True
    PLOT_HISTS = False
    PLOT_RETAINED_SHOTS = False
    SAVE_FLUCT_CALIB = False

    # Count atoms

    atom_numbers_shot = dict.fromkeys(uj_vals)
    hists_shot = dict.fromkeys(uj_vals)

    for uj in recentered_data:
        lattice_axis = recentered_data[uj].columns[0]
        atom_numbers_shot[uj] = pd.Series(index=recentered_data[uj][lattice_axis].index, dtype=float)
        for shot in recentered_data[uj][lattice_axis].index:
            atom_numbers_shot[uj][shot] = len(recentered_data[uj][lattice_axis][shot])
            # Check that all three axes contain same number of atoms:
            assert all([atom_numbers_shot[uj][shot] == len(recentered_data[uj][lattice_axis][shot]) for lattice_axis in recentered_data[uj].columns])

    mean_atom_numbers_shot = {uj: np.mean([shot for shot in atom_numbers_shot[uj]]) for uj in uj_vals}

    # Create histograms:
    for uj in recentered_data:
        hists_shot[uj] = bh.Histogram(bh.axis.Regular(
                                max([int(atom_numbers_shot[uj].max()) for uj in uj_vals]),
                                0,
                                max([int(atom_numbers_shot[uj].max()) for uj in uj_vals])
                                ))
        hists_shot[uj].fill(atom_numbers_shot[uj])

    # Compare with calibration:
    def calibrate_lattice_atom_number(uj):
        return sum(val * uj ** key for key, val in lattice_atom_number_calibration.items())

    calibrated_atom_numbers = {uj: calibrate_lattice_atom_number(uj) for uj in uj_vals}

    # Fit Gaussians:
    def one_d_gaussian_on_axis(x, amp, x0, sigma):
        """"Fitting funciton for 1D Gaussiians."""
        return amp * np.exp(-np.power((x - x0)/sigma, 2)/2)

    fits = dict.fromkeys(recentered_data)

    for uj in recentered_data:

        # Set fit sigma to include 30% fluctuations around mean:

        """
        Fluctuation percentage cutoff = 30%.
        Let's assume cutoff ~ 2 * sigma = 95% for that.
        Then: µ * 0.7 = µ - 2 * sigma => sigma = µ * 0.15.
        Or for arbitrary fluctuation percentage cutoffs:
        sigma = µ * FLUCT_PERC / 2
        """

        sigma_guess = mean_atom_numbers_shot[uj] * FLUCT_PERC/100 / 2

        if SELECT_FOR_CALIB:
            x0_guess = calibrated_atom_numbers[uj]
        else:
            peaks, _ = find_peaks(
                hists_shot[uj].view(),
                height=np.max(hists_shot[uj].view()),
                )
            x0_guess = hists_shot[uj].axes[0].centers[peaks[0]]

        initial_guess = (
            np.amax(hists_shot[uj].view()),
            # + fit_ranges[lattice_axis][order][mcp_axis]['min_val']],
            x0_guess,
            sigma_guess
        )
        # Set bounds for the fit:
        fit_stringency = 0.01
        fitted = False
        while not fitted:
            bounds = ([
                0.8 * np.amax(hists_shot[uj].view()),
                x0_guess - 2,
                (1-fit_stringency)*sigma_guess
            ], [
                1.2 * np.amax(hists_shot[uj].view()),
                x0_guess + 2,
                (1+fit_stringency)*sigma_guess]
            )
            try:
                fits[uj], pcov = curve_fit(
                    one_d_gaussian_on_axis,
                    hists_shot[uj].axes[0].centers,
                    hists_shot[uj].view(),
                    p0=initial_guess,
                    bounds=bounds
                )
                fitted = True
            except RuntimeError:
                fit_stringency += 0.01

    # Calculate number of retained shots:
    retained_shots_numbers[FLUCT_PERC] = dict.fromkeys(recentered_data)
    retained_shots[FLUCT_PERC] = dict.fromkeys(recentered_data)

    for uj in retained_shots_numbers[FLUCT_PERC]:
        # Retain two sigma:
        retained_shots_numbers[FLUCT_PERC][uj] = int(hists_shot[uj][
            math.floor(fits[uj][1]-2*fits[uj][2]):math.ceil(fits[uj][1]+2*fits[uj][2])
            ].sum())
        retained_shots[FLUCT_PERC][uj] = atom_numbers_shot[uj].loc[np.where(
            np.logical_and(
                atom_numbers_shot[uj] > math.floor(fits[uj][1]-2*fits[uj][2]),
                atom_numbers_shot[uj] < math.ceil(fits[uj][1]+2*fits[uj][2])
                )
            )]


    # Plot

    plot_color_str = """55,126,184
        228,26,28
        255,127,0"""

    def make_plot_colors(color_str):
        return [tuple([int(i)/255 for i in color_str.split('\n')[j].split(',')])
                for j in range(len(color_str.split('\n')))]

    plot_colors = make_plot_colors(plot_color_str)

    if PLOT_HISTS in (FLUCT_PERC, True):

        fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)

        for idx, ax in enumerate(axes.flatten()):
            # Leave upper left field empty (11 data series for 12 fields):
            if idx > 0:
                idx -= 1
                # Plot histogram of detected atom numbers:
                ax.plot(
                    hists_shot[uj_vals[idx]].axes[0].centers,
                    hists_shot[uj_vals[idx]].view(),
                    linestyle='-',
                    linewidth=2,
                    color=plot_colors[0],
                    label=f'Mean: {mean_atom_numbers_shot[uj_vals[idx]]:.0f}'
                    )
                # Plot calibration target:
                ax.axvline(
                    x=calibrated_atom_numbers[uj_vals[idx]],
                    color=plot_colors[1],
                    linestyle='-',
                    linewidth=2,
                    label=f'Calib: {int(calibrated_atom_numbers[uj_vals[idx]])}'
                    )
                # Plot fit:
                ax.plot(
                    hists_shot[uj_vals[idx]].axes[0].centers,
                    one_d_gaussian_on_axis(hists_shot[uj_vals[idx]].axes[0].centers, *fits[uj_vals[idx]]),
                    linestyle='-',
                    linewidth=2,
                    color=plot_colors[2],
                    label=f'Fit ({FLUCT_PERC}%): {fits[uj_vals[idx]][1]:.0f}'
                    )
                if uj_vals[idx].is_integer():
                    ax.set_title(f'U/J = {int(uj_vals[idx])}: {retained_shots_numbers[FLUCT_PERC][uj_vals[idx]]}/{len(atom_numbers_shot[uj_vals[idx]])} Shots')
                else:
                    ax.set_title(f'U/J = {uj_vals[idx]}: {retained_shots_numbers[FLUCT_PERC][uj_vals[idx]]}/{len(atom_numbers_shot[uj_vals[idx]])} Shots')
                if idx in (0, 3, 7):
                    ax.set_ylabel('Counts')
                if idx > 6:
                    ax.set_xlabel('Atom number')
                ax.legend()
            else:
                ax.set_visible(False)
        plt.suptitle('Detected Atom Number Histograms')
        plt.show()

#%%

rel_fluct_perc = dict.fromkeys(fluct_perc_vals)
eff_std = dict.fromkeys(fluct_perc_vals)

fluct_perc_vals_plot = sorted(fluct_perc_vals)
for FLUCT_PERC in fluct_perc_vals_plot:
    retained_shots_numbers_plot = [retained_shots_numbers[FLUCT_PERC][uj] for uj in sorted(uj_vals)]
    eff_std[FLUCT_PERC] = [retained_shots[FLUCT_PERC][uj].std() for uj in sorted(uj_vals)]
    rel_fluct_perc[FLUCT_PERC] = [100 * retained_shots[FLUCT_PERC][uj].std() / retained_shots[FLUCT_PERC][uj].mean() for uj in sorted(uj_vals)]


if PLOT_RETAINED_SHOTS:

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    for FLUCT_PERC in fluct_perc_vals_plot:

        ax1.plot(
            sorted(uj_vals),
            retained_shots_numbers_plot,
            label=f'{FLUCT_PERC}% - Mean: {int(np.mean(retained_shots_numbers_plot))}'
            )

        ax2.plot(
            sorted(uj_vals),
            eff_std[FLUCT_PERC],
            label=f'{FLUCT_PERC}% - Mean: {np.mean(eff_std[FLUCT_PERC]):.1f}'
            )

        ax3.plot(
            sorted(uj_vals),
            rel_fluct_perc[FLUCT_PERC],
            label=f'{FLUCT_PERC}% - Mean: {np.mean(rel_fluct_perc[FLUCT_PERC]):.1f}%'
            )

    ax1.set_ylabel('Number of shots')
    ax1.set_title('Number of retained shots for post-selection fluctuation percentages')
    ax1.legend(loc="upper right")
    ax1.grid()

    ax2.set_ylabel(r'$\Delta N$')
    ax2.set_title('Standard deviations of atom number for retained shots')
    ax2.legend(loc="upper right")
    ax2.grid()

    ax3.set_xlabel('U/J')
    ax3.set_ylabel(r'$\Delta N / N\ [\%]$')
    ax3.set_title('Relative atom number fluctuations for retained shots')
    ax3.legend(loc="upper right")
    ax3.grid()

    plt.tight_layout()
    plt.show()

#%% Fit relative fluctuation (cutoff_perc) to post-select more precisely:

def poly_two(x, a, b, c): return a + x * b + x**2 * c

popt, pcov = curve_fit(
    poly_two,
    sorted(fluct_perc_vals),
    [np.mean(rel_fluct_perc[FLUCT_PERC]) for FLUCT_PERC in sorted(fluct_perc_vals)],
    p0=[0, 0.5, 0.1]
    )

plt.figure()
plt.errorbar(
    sorted(fluct_perc_vals),
    [np.mean(rel_fluct_perc[FLUCT_PERC]) for FLUCT_PERC in sorted(fluct_perc_vals)],
    yerr=[np.std(rel_fluct_perc[FLUCT_PERC]) for FLUCT_PERC in sorted(fluct_perc_vals)],
    label='Relative Fluctuations',
    color=plot_colors[0]
    )
plt.plot(
    sorted(fluct_perc_vals),
    [poly_two(x, *list(popt)) for x in sorted(fluct_perc_vals)],
    label=rf'Poly Fit: {popt[-1]:.3f} $\times x^2$ + {popt[-2]:.3f} $\times$ x + {popt[-3]:.3f}',
    color=plot_colors[1],
    linewidth=4
    )
plt.xlabel('Cutoff Fluctuations [%]')
plt.ylabel(r'$\Delta N / N\ [\%]$')
plt.title('Relative Fluctuations as a function of post-selection fluctuation cutoff')
plt.legend(loc='upper left')
plt.grid()
plt.show()

# Save fit result:
def rel_fluct(fluct_cutoff): return poly_two(fluct_cutoff, *popt)
if SAVE_FLUCT_CALIB:
    with open('rel_fluct_calib.pickle', 'wb') as outfile:
        pickle.dump(popt, outfile)

# Fit inverse function:
popt_inv, pcov_inv = curve_fit(
    poly_two,
    [np.mean(rel_fluct_perc[FLUCT_PERC]) for FLUCT_PERC in sorted(fluct_perc_vals)],
    sorted(fluct_perc_vals),
    p0=[0, 0.5, 0.1]
    )

def fluct_cutoff(rel_fluct): return poly_two(rel_fluct, *popt_inv)


#%% Bin cutoff fluctuations for N number of shots:

N_min = 100

mean_retained_shots_numbers = {FLUCT_PERC: np.mean([retained_shots_numbers[FLUCT_PERC][uj] for uj in uj_vals]) for FLUCT_PERC in sorted(fluct_perc_vals)}

bins = dict()

shot_ctr = 0

for FLUCT_PERC in sorted(fluct_perc_vals):
    shot_ctr += mean_retained_shots_numbers[FLUCT_PERC]
    if shot_ctr > N_min:
        bins[FLUCT_PERC] = shot_ctr
        shot_ctr = 0

plt.figure()
plt.plot(
    sorted(bins.keys()),
    [bins[i] for i in sorted(bins.keys())],
    color=plot_colors[0],
    label='Number of shots in bin'
    )
plt.axhline(
    N_min,
    color='k',
    linestyle='-',
    linewidth=2,
    label='Minimum shots per bin'
    )
for bin_edge in sorted(bins.keys()):
    if bin_edge < sorted(bins.keys())[-1]:
        plt.axvline(
            x=bin_edge,
            color=plot_colors[1],
            linestyle='-',
            linewidth=1
            )
    else:
        plt.axvline(
            x=bin_edge,
            color=plot_colors[1],
            linestyle='-',
            linewidth=1,
            label='Bins'
            )
plt.xlabel('Cutoff Fluctuations [%]')
plt.ylabel('Number of shots in bin')
plt.title(f'Binning the cutoff fluctuations for minimum of {N_min} retained shots per bin')
plt.legend()
plt.show()

#%% Post-select for effective relative fluctuation target and plot atom number variations:


# for rel_fluct_target in sorted(bins.keys()):
rel_fluct_target = sorted(bins.keys())[0]  # % Rel. Fluct, not cutoff!
fluct_perc_for_target = fluct_cutoff(rel_fluct_target)

def post_select(rel_fluct_target, ctr_val=calibrated_atom_numbers):


    fluct_perc_for_target = fluct_cutoff(rel_fluct_target)

    ps_indices = dict.fromkeys(sorted(uj_vals))

    for uj in sorted(uj_vals):


        calib_mean = ctr_val[uj]
        ps_indices[uj] = np.where(abs(atom_numbers_shot[uj] - calib_mean) / atom_numbers_shot[uj] < (fluct_perc_for_target/100))[0]

    return ps_indices

post_selection = {uj: atom_numbers_shot[uj][post_select(rel_fluct_target)[uj]] for uj in sorted(uj_vals)}
eff_std = {uj: post_selection[uj].std() for uj in post_selection}
eff_rel_fluct_perc = {uj: 100 * post_selection[uj].std() / post_selection[uj].mean() for uj in post_selection}

fig, axes = plt.subplots(4, 3, figsize=(16, 9))

for idx, ax in enumerate(axes.flatten()):
    # Leave upper left field empty (11 data series for 12 fields):
    if idx == 2:
        ax.set_visible(False)
        continue
    elif idx > 2:
        idx -= 1
    uj = sorted(uj_vals)[idx]
    ax.plot(
        [idx for idx, _ in enumerate(post_selection[uj].index)],
        post_selection[uj].values,
        color=plot_colors[0],
        label=rf'Std: {eff_std[uj]:.1f} - $\Delta N / N\ : ${eff_rel_fluct_perc[uj]:.1f} %'
        )
    ax.axhline(
        calibrated_atom_numbers[uj],
        color=plot_colors[1],
        linestyle='-',
        linewidth=2,
        label=f'Calib: {int(calibrated_atom_numbers[uj_vals[idx]])}'
        )
    ax.grid()
    if uj_vals[idx].is_integer():
        ax.set_title(f'U/J = {int(uj_vals[idx])}: {len([idx for idx, _ in enumerate(post_selection[uj].index)])} Shots, Mean: {np.mean(post_selection[uj].values):.0f}')
    else:
        ax.set_title(f'U/J = {uj_vals[idx]}: {max([idx for idx, _ in enumerate(post_selection[uj].index)])} Shots, Mean: {np.mean(post_selection[uj].values):.0f}')
    if idx > 7:
        ax.set_xlabel('Shots')
    if idx in (0, 2, 5, 8):
        ax.set_ylabel('Atom number')
    ax.legend(loc='lower left')
plt.suptitle(f'Detected Atom Numbers for Relative Fluctuations of {np.mean([eff_rel_fluct_perc[uj] for uj in sorted(uj_vals)]):.1f} % (Post selection cutoff: {fluct_perc_for_target:.1f}%)')
plt.tight_layout()
plt.show()

#%% Post-selection

post_selected_data = dict.fromkeys(sorted(bins.keys()))

for rel_fluct_target in tqdm(sorted(bins.keys()), desc='Post Selection'):

    post_selected_data[rel_fluct_target] = dict.fromkeys(sorted(uj_vals))
    for uj in sorted(uj_vals):

        post_selected_data[rel_fluct_target][uj] = recentered_data[uj].loc[post_select(rel_fluct_target, ctr_val=calibrated_atom_numbers)[uj]]

#%% Calculate variance:

k_min = 0
k_max = 0.15

def eta(uj):
    return calibrate_lattice_atom_number(uj)/5000

bec_peak_atom_numbers = dict.fromkeys(post_selected_data)
shot_atom_numbers = dict.fromkeys(post_selected_data)
variance = dict.fromkeys(post_selected_data)
relative_fluctuations = dict.fromkeys(post_selected_data)

# Count number of atoms in BEC and in entire shot:
for rel_fluct_target in tqdm(post_selected_data, desc='Calculating variance'):
    bec_peak_atom_numbers[rel_fluct_target] = dict.fromkeys(post_selected_data[rel_fluct_target])
    relative_fluctuations[rel_fluct_target] = dict.fromkeys(post_selected_data[rel_fluct_target])
    shot_atom_numbers[rel_fluct_target] = dict.fromkeys(post_selected_data[rel_fluct_target])
    variance[rel_fluct_target] = dict.fromkeys(post_selected_data[rel_fluct_target])
    bec_peak_atom_numbers[rel_fluct_target] = dict.fromkeys(post_selected_data[rel_fluct_target])
    for uj in post_selected_data[rel_fluct_target]:
        bec_peak_atom_numbers[rel_fluct_target][uj] = dict.fromkeys(post_selected_data[rel_fluct_target][uj].index.dropna())
        relative_fluctuations[rel_fluct_target][uj] = dict.fromkeys(post_selected_data[rel_fluct_target][uj].index.dropna())
        shot_atom_numbers[rel_fluct_target][uj] = dict.fromkeys(post_selected_data[rel_fluct_target][uj].index.dropna())
        for run in post_selected_data[rel_fluct_target][uj]['k_h'].index.dropna():

            indices = dict.fromkeys(['k_h', 'k_m45', 'k_p45'])

            for axis in indices:
                indices[axis] = (k_min < abs(post_selected_data[rel_fluct_target][uj][axis][run])) * (abs(post_selected_data[rel_fluct_target][uj][axis][run]) < k_max)
            indices_all = np.where(indices['k_h'] * indices['k_m45'] * indices['k_p45'])
            assert len(list(post_selected_data[rel_fluct_target][uj]['k_h'][run][indices_all])) == len(list(post_selected_data[rel_fluct_target][uj]['k_m45'][run][indices_all])) == len(list(post_selected_data[rel_fluct_target][uj]['k_p45'][run][indices_all]))
            bec_peak_atom_numbers[rel_fluct_target][uj][run] = len(list(post_selected_data[rel_fluct_target][uj]['k_h'][run][indices_all]))
            assert len(list(post_selected_data[rel_fluct_target][uj]['k_h'][run])) == len(list(post_selected_data[rel_fluct_target][uj]['k_m45'][run])) == len(list(post_selected_data[rel_fluct_target][uj]['k_p45'][run]))
            shot_atom_numbers[rel_fluct_target][uj][run] = len(list(post_selected_data[rel_fluct_target][uj]['k_h'][run]))


    # Calculate variance and normalize:
    for uj in post_selected_data[rel_fluct_target]:
        variance[rel_fluct_target][uj] = np.var(list(bec_peak_atom_numbers[rel_fluct_target][uj].values()))
        relative_fluctuations[rel_fluct_target][uj] = variance[rel_fluct_target][uj]/np.mean(list(shot_atom_numbers[rel_fluct_target][uj].values())) / eta(float(uj))

#%% Plot variance

fig, ax1 = plt.subplots(figsize=(16, 9))
ax2 = ax1.twinx()

def make_plot_colors(color_str):
    return [tuple([int(i)/255 for i in color_str.split('\n')[j].split(',')])
            for j in range(len(color_str.split('\n')))]

if 7.5 in uj_vals:
    uj_vals.remove(7.5)

for rel_fluct_target in (0.8, 2.0, 4.0, 6.0, 9.5):

    variance_plot = [variance[rel_fluct_target][uj] for uj in sorted(uj_vals)]
    relative_fluctuations_plot = [relative_fluctuations[rel_fluct_target][uj] for uj in sorted(uj_vals)]
    bec_peak_atom_numbers_plot = [np.mean(list(bec_peak_atom_numbers[rel_fluct_target][uj].values())) for uj in sorted(uj_vals)]
    atom_numbers_plot = [np.mean(list(shot_atom_numbers[rel_fluct_target][uj].values())) for uj in sorted(uj_vals)]
    # Plot fluctuations:
    real_atom_nb_k = np.mean(np.mean([list(shot_atom_numbers[rel_fluct_target][uj].values()) for uj in sorted(uj_vals)]))/1000
    fluct_std_perc = rel_fluct(rel_fluct_target)#100*np.mean([np.std(list(shot_atom_numbers[rel_fluct_target][uj].values()))/np.mean(list(shot_atom_numbers[rel_fluct_target][uj].values())) for uj in sorted(uj_vals) for uj in sorted(uj_vals)])
    # file_nb =
    plot_label_var = f'{fluct_std_perc:.2f}% fluct'
    plot_label_bec = f'{fluct_std_perc:.2f}% fluct'
    plot_label_rf = f'{fluct_std_perc:.2f}% fluct'
    plot_label_sf = f'{fluct_std_perc:.2f}% fluct'
    plot_label_an = f'{fluct_std_perc:.2f}% fluct'
    # ax1.plot(sorted(uj_vals), variance_plotrelative_fluctuations, color=plot_colors, label=plot_label_var)#, marker='*')
    # ax2.scatter(sorted(uj_vals), bec_peak_atom_numbers_plot, color=plot_colors[rel_fluct_target], label=plot_label_bec, marker='+')
    ax2.scatter(sorted(uj_vals), atom_numbers_plot, label=plot_label_an)
    ax1.plot(sorted(uj_vals), relative_fluctuations_plot, label=plot_label_rf)
# ax2.plot(
#     sorted(uj_vals),
#     [calibrate_lattice_atom_number(uj)*n_factor for uj in sorted(uj_vals)],
#     color=plot_colors,
#     linestyle=':',
#     linewidth=0.75,
#     label='Fit from calibration'
#     )
ax1.grid()
ax1.set_xlabel('Lattice depth [U/J]')
ax1.set_ylabel(r'$(\Delta N_{BEC})^2\ /\ N$')
# ax1.set_ylabel(r'$(\Delta N_{BEC})^2$')
ax2.set_ylabel('Atom Numbers')
plt.title('Relative atom number fluctuations in the BEC center for ' + r'$k_{max}$' + ' = ' + f"{k_max} " + r'$k_d$')
# plt.title('variance of the atom number in the BEC for ' + r'$k_{max}$' + ' = ' + f"{k_max} " + r'$k_d$')
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='right', bbox_to_anchor=(0.5, 0.1, 0.4, 0.8))
plt.tight_layout()
plt.show()

#%%

flat_zone_idxs = uj_vals[:np.where(np.array(uj_vals)==20.0)[0][0]]
fluct_zone_idxs = uj_vals[len(flat_zone_idxs):np.where(np.array(uj_vals)==25.0)[0][0]]

flat_zone_flucts = dict.fromkeys(post_selected_data)
fluct_zone_flucts = dict.fromkeys(post_selected_data)
fluct_param = dict.fromkeys(post_selected_data)

for rel_fluct_target in sorted(post_selected_data.keys()):
    flat_zone_flucts[rel_fluct_target] = np.mean([relative_fluctuations[rel_fluct_target][uj] for uj in sorted(flat_zone_idxs)])
    fluct_zone_flucts[rel_fluct_target] = np.mean([relative_fluctuations[rel_fluct_target][uj] for uj in sorted(fluct_zone_idxs)])
    fluct_param[rel_fluct_target] = fluct_zone_flucts[rel_fluct_target] - flat_zone_flucts[rel_fluct_target]

plt.figure()
plt.scatter(
    [rel_fluct(fluct_cutoff) for fluct_cutoff in sorted(post_selected_data.keys())],
    [fluct_param[rel_fluct_target] for rel_fluct_target in sorted(post_selected_data.keys())],
    color=plot_colors[0],
    label=r'$\left[(\Delta N_{BEC})^2\ /\ N\right]\ |_{15 \leq \frac{U}{J} \leq 25} - \left[(\Delta N_{BEC})^2\ /\ N\right]\ |_{0 \leq \frac{U}{J} \leq 15}$'
    )
plt.axvline(
    x=np.sqrt(1/5e3)*100,
    color=plot_colors[1],
    linestyle='-',
    linewidth=2,
    label=r'Shot noise ~ $\sqrt{\frac{1}{5000}} \simeq 1.4$%'
    )
plt.axvline(
    x=rel_fluct(5.234343434343434),
    color=plot_colors[2],
    linestyle='-',
    linewidth=2,
    label='"Visibility limit of BEC fluctuations" ' + r'$\simeq$' + '{:.1f} %'.format(rel_fluct(5.234343434343434))
    )
plt.xlabel(r'$\frac{\Delta N}{N}$ [%]')
plt.ylabel(r'$\Delta\ \left[ \frac{(\Delta N_{BEC})^2}{N} \right]$')
plt.title('Comparison of fluctuation magnitudes around transition')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
