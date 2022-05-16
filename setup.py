#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:14:34 2022

General setup functions for the analysis of BEC fluctuations. Set datapaths
on different machines and load calibrations and data.

@author: jp
"""

# Standard library imports:
import socket
import os
import pickle
import pandas as pd
import boost_histogram as bh

# Local imports:
from helper_functions import multiproc_list
from mcpmeas.load_recentered_data import load_recentered_data


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
    # Server:
    elif hostname == 'zito':
        data_basepath = os.path.join(
            home,
            'Documents/recentered_data/indiv_all'
            )
        figure_savepath = os.path.join(
            home,
            'Pictures/data_analysis/bec-fluctuations',
            'calibrate_rel_fluct_from_cutoff.png'
            )
        # Load atom number calibration:
        atom_number_calib_path = os.path.join(
            home,
            'Documents/data_analysis/bec-fluctuations',
            'lattice_atom_number_calibration_data.pickle'
            )
        with open(atom_number_calib_path, 'rb') as infile:
            lattice_atom_number_calibration = pickle.load(infile)
            
    # JupyterHub:
    elif hostname == 'b83123837503':
        data_basepath = os.path.join(
            home,
            'data/recentered_data/indiv_all'
            )
        figure_savepath = os.path.join(
            home,
            'Pictures/data_analysis/bec-fluctuations',
            'calibrate_rel_fluct_from_cutoff.png'
            )
        # Load atom number calibration:
        atom_number_calib_path = os.path.join(
            home,
            'work/data_analysis/bec-fluctuations',
            'lattice_atom_number_calibration_data.pickle'
            )
        with open(atom_number_calib_path, 'rb') as infile:
            lattice_atom_number_calibration = pickle.load(infile)

    return data_basepath, figure_savepath, lattice_atom_number_calibration

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

def get_atom_numbers_all_shots(recentered_data, uj_vals):
    
    """Count all the atoms in all the shots for all values of U/J and return
    them in a DataFrame."""
    
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
        
    return atom_numbers_all_shots
