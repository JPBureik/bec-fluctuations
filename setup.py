#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:14:34 2022

General setup functions for the analysis of BEC fluctuations. Set datapaths
on different machines and load calubrations and data.

@author: jp
"""

# Standard library imports:
import socket
import os
import pickle

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
