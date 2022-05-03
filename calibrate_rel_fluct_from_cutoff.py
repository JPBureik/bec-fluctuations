#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:31:51 2022

@author: jp

Calibration of relative fluctuations of atom numbers in post-selection.

Calibrate the relative fluctuations for the shots that are retained in post-
selection when limiting the shots to those with atom numbers in a window of a
certain percentage of a central number. This corresponds to the usual post-
selection procedure for our experimental MCP data, with the usual value of the
cutoff fluctuation percentage being 30% (PS_CUTOFF_MAX_PERC). The central
number can be set by the calibration of the atom numbers as a function of U/J
for 5k atoms in the trap (USE_ATOM_NUMBER_CALIB_UJ); otherwise the mean atom
number will be used for each dataset. The resulting relative fluctuations (in
%) can be plotted (PLOT_FLUCT_CALIB) and saved (SAVE_FLUCT_CALIB_*).
"""

# Standard library imports:
import socket
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import boost_histogram as bh
from matplotlib import pyplot as plt
import pickle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math

# Local imports:
from helper_functions import multiproc_list
from plot_colors import get_plot_colors
from mcpmeas.load_recentered_data import load_recentered_data

""" ---------- INPUT ---------- """
PS_CUTOFF_MAX_PERC = 30  # %
USE_ATOM_NUMBER_CALIB_UJ = True
PLOT_FLUCT_CALIB = True
SAVE_FLUCT_CALIB_FIG = True
SAVE_FLUCT_CALIB_FCTN = False

#%% Setup

def setup():

    """Set path to data directory and figure save path.
    Load atom number calibration."""
    hostname = socket.gethostname()
    home = os.path.expanduser('~')
    # Office:
    if hostname == 'catarella':
        data_basepath = os.path.join(
            home,
            'Documents/data/recentered_data/indiv_all'
            )
        figure_savepath = os.path.join(
            home,
            'Pictures/prog/bec-fluctuations',
            'calibrate_rel_fluct_from_cutoff.png'
            )
        # Load atom number calibration:
        atom_number_calib_path = os.path.join(
            home,
            'Documents/prog/work/bec-fluctuations',
            'lattice_atom_number_calibration_data.pickle'
            )
        with open(atom_number_calib_path, 'rb') as infile:
            lattice_atom_number_calibration = pickle.load(infile)
    # Home:
    elif hostname == 'jp-MS-7C02':
        data_basepath = os.path.join(
            home,
            'Documents/data/recentered_data/indiv_all'
            )
        figure_savepath = os.path.join(
            home,
            'Pictures/prog/bec-fluctuations',
            'calibrate_rel_fluct_from_cutoff.png'
            )
        # Load atom number calibration:
        atom_number_calib_path = os.path.join(
            home,
            'Documents/prog/work/bec-fluctuations',
            'lattice_atom_number_calibration_data.pickle'
            )
        with open(atom_number_calib_path, 'rb') as infile:
            lattice_atom_number_calibration = pickle.load(infile)

    return data_basepath, figure_savepath, lattice_atom_number_calibration

#%% Load data:

def load_data(data_basepath):

    """Load recentered datasets from data_basepath using multiprocessing on
    the standard load function from the mcpmeas module and extract U/J values
    from filenames."""

    def uj_from_filename(f):
        """String formatting helper function."""
        return float(f.split('uj')[1].split('_all')[0].replace('p','.'))

    filenames = sorted(
        [filename for filename in os.walk(data_basepath)][0][2]
        )
    datapaths = sorted(
        [os.path.join(data_basepath, filename) for filename in filenames]
        )
    uj_vals = sorted(
        [uj_from_filename(filename) for filename in filenames]
        )

    # Load recentered data:
    recentered_data_list = multiproc_list(
        datapaths,
        load_recentered_data,
        show_pbar=True,
        desc='Loading recentered data'
        )
    # Explicitly track U/J values with datasets:
    recentered_data = {uj: recentered_dataset for recentered_dataset, uj in
                       zip(recentered_data_list, uj_vals)}

    return recentered_data, uj_vals

#%% Calibrate relative fluctuations from cutoff percentage:

def calibrate_rel_fluct_from_cutoff(
        recentered_data,
        uj_vals,
        lattice_atom_number_calibration,
        use_atom_number_calib_uj=USE_ATOM_NUMBER_CALIB_UJ,
        ps_cutoff_max_perc=PS_CUTOFF_MAX_PERC
        ):

    """Calibration of the relative fluctuations resulting from post-selection
    of shots around a central value (can be the one calculated for 5k in trap
    as a function of U/J) up to a given cutoff value that is equal to +/- a
    certain percentage of the central value. For a fixed (set of) central N,
    determine the dependency rel_fluct(cutoff), where rel_fluct = std(n)/n
    with n the remaining shots after post-selection around N limited by
    cutoff, where cutoff is given as a percentage of N."""

    # Define lattice atom number calibration function:
    def calibrate_lattice_atom_number(uj):
        return sum(
            val * uj ** key
            for key, val in lattice_atom_number_calibration.items()
            )

    # Determine calibrated atom number for 5k for all U/J's:
    calibrated_atom_numbers = pd.Series({
        uj: calibrate_lattice_atom_number(uj)
        for uj in uj_vals
        })

    # Count the total atom number for all shots in all datasets:
    atom_numbers_all_shots = pd.DataFrame(
        data={uj:
            pd.Series(
            data=[
                recentered_data[uj]['k_m45'][shot].shape[0]
                for shot in recentered_data[uj]['k_h'].index
                ],
            dtype=float
            )
            for uj in sorted(uj_vals)},
        index=range(max([
            recentered_data[uj]['k_m45'].shape[0]
            for uj in uj_vals
            ])),
        columns=sorted(uj_vals),
        dtype=float
        )

    # Create and fill histograms:
    hists_shot = {
        uj: bh.Histogram(bh.axis.Regular(
            int(atom_numbers_all_shots.max().max()),
            0,
            int(atom_numbers_all_shots.max().max())
            )).fill(atom_numbers_all_shots[uj])
        for uj in uj_vals
        }

    # Create 100 sample values below ps_cutoff_max_perc:
    cutoff_perc_vals = np.linspace(0.1, ps_cutoff_max_perc, int(1e3))

    # Compute mean over all shots for each dataset:
    mean_atom_numbers_all_shots = atom_numbers_all_shots.mean(axis=0)

    # Define central atom number around which to post-select:
    if use_atom_number_calib_uj:
        ctr_atom_number = calibrated_atom_numbers
    else:
        ctr_atom_number = mean_atom_numbers_all_shots

    # Count the number of shots remaining for each U/J when limiting the
    # recentered data to shots with a detected atom number in-between
    # (1 - cutoff_perc, 1 + cutoff_perc) * central number:
    atom_numbers_retained_shots = pd.DataFrame.from_dict(
        data={cutoff_perc: {
            uj:
                atom_numbers_all_shots[uj].loc[np.where(
                    np.logical_and(
                        atom_numbers_all_shots[uj] >= math.floor(
                            ctr_atom_number[uj]
                            * (1 - cutoff_perc / 100)
                        ),
                        atom_numbers_all_shots[uj] < math.ceil(
                            ctr_atom_number[uj]
                            * (1 + cutoff_perc / 100)
                        )
                    )
                )]
            for uj in uj_vals}
        for cutoff_perc in cutoff_perc_vals},
        orient='index',
        columns=uj_vals,
        dtype=object
        )

    # Compute mean over retained shots for each dataset:
    mean_atom_numbers_retained_shots = atom_numbers_retained_shots.applymap(
                                                                    np.mean)

    # Compute Standard Deviation and relative fluctuations for any cutoff_perc:
    std_retained_shots = atom_numbers_retained_shots.applymap(np.std)
    rel_fluct_retained_shots = std_retained_shots.divide(
                                    mean_atom_numbers_retained_shots
                                    )
    rel_fluct_perc_retained_shots = rel_fluct_retained_shots.multiply(1e2)

    # Compute mean and std over all datasets for each value of cutoff_perc:
    mean_rel_fluct_perc_retained_shots = rel_fluct_perc_retained_shots.mean(
                                                                       axis=1)
    std_rel_fluct_perc_retained_shots = rel_fluct_perc_retained_shots.std(
                                                                       axis=1)

    return (
        cutoff_perc_vals,
        mean_rel_fluct_perc_retained_shots,
        std_rel_fluct_perc_retained_shots
        )

#%% Fit and plot relative fluctuation from cutoff values:

def fit_and_plot_rel_fluct_from_cutoff(
        cutoff_perc_vals,
        mean_rel_fluct_perc_retained_shots,
        std_rel_fluct_perc_retained_shots,
        figure_savepath
        ):

    """Fit the relative_fluctuations with a polynomial of order 2 and plot the
    result."""

    # Define fitting function:
    def poly_two(x, a, b, c): return a * x**2 + b * x + c

    # Helper function to display correct sign for fitting function in plot:
    def get_sign_str(real_number):
        if np.sign(real_number) == 1:
            return '+'
        else:
            return '-'
        
    # Load colors for plotting:
    plot_colors = get_plot_colors('qualitative', 3, name='Set1')

    popt, pcov = curve_fit(
        poly_two,
        cutoff_perc_vals,
        mean_rel_fluct_perc_retained_shots,
        p0=[0, 0.5, 0.1]
        )
    
    # Plot:
    if PLOT_FLUCT_CALIB:
        plot_label = ' '.join([
            f'Poly Fit: {popt[0]:.3f}',
            r'$\times\ F_{\mathrm{PS}}^2$ +',
            f'{popt[1]:.3f}',
            r'$\times\ F_{\mathrm{PS}}$',
            f'{get_sign_str(popt[2])} {abs(popt[2]):.3f}'
            ])
    
        plt.figure(figsize=(7, 3))
        plt.errorbar(
            cutoff_perc_vals,
            mean_rel_fluct_perc_retained_shots,
            yerr=std_rel_fluct_perc_retained_shots,
            label='Relative Fluctuations',
            color=plot_colors[1],
            linewidth=0.5,
            elinewidth=0.5
            )
        plt.plot(
            cutoff_perc_vals,
            [poly_two(x, *list(popt)) for x in cutoff_perc_vals],
            label=plot_label,
            color=plot_colors[0],
            linewidth=1.5,
            linestyle='--'
            )
        plt.xlabel(r'Post-Selection Cutoff Fluctuations $F_{\mathrm{PS}}$ [%]')
        plt.ylabel(r'$\Delta N / N\ [\%]$')
        plt.title('Relative Fluctuations as a function of Post-Selection '
                  + 'Fluctuation Cutoff')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    if SAVE_FLUCT_CALIB_FIG:
        plt.savefig(figure_savepath, dpi='figure')

        # Save fit result:
        # def rel_fluct_perc(fluct_cutoff): return poly_two(fluct_cutoff, *popt)
        # if SAVE_FLUCT_CALIB:
        #     with open('rel_fluct_perc_calib.pickle', 'wb') as outfile:
        #         pickle.dump(popt, outfile)

        # # Fit inverse function:
        # popt_inv, pcov_inv = curve_fit(
        #     poly_two,
        #     [np.mean(rel_fluct_perc[cutoff_perc]) for cutoff_perc in sorted(cutoff_perc_vals)],
        #     sorted(cutoff_perc_vals),
        #     p0=[0, 0.5, 0.1]
        #     )

        # def fluct_cutoff(rel_fluct_perc): return poly_two(rel_fluct_perc, *popt_inv)


#%% Execution:

if __name__ == '__main__':

    data_basepath, figure_savepath, lattice_atom_number_calibration = setup()
    recentered_data, uj_vals = load_data(data_basepath)
    (
     cutoff_perc_vals,
     mean_rel_fluct_perc_retained_shots,
     std_rel_fluct_perc_retained_shots
     ) = calibrate_rel_fluct_from_cutoff(
         recentered_data,
         uj_vals,
         lattice_atom_number_calibration,
         use_atom_number_calib_uj=USE_ATOM_NUMBER_CALIB_UJ,
         ps_cutoff_max_perc=PS_CUTOFF_MAX_PERC
         )
    fit_and_plot_rel_fluct_from_cutoff(
        cutoff_perc_vals,
        mean_rel_fluct_perc_retained_shots,
        std_rel_fluct_perc_retained_shots,
        figure_savepath
        )
    if SAVE_FLUCT_CALIB_FCTN:
        ...

