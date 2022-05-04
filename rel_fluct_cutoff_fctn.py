#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:13:38 2022

Function definitions to infer the relative fluctuation percentage from the
post-selection cutoff fluctuations and vice versa.
Based on the calibration results.

@author: jp
"""

# Standard library imports:
import pickle
import numpy as np

# Local imports:
from helper_functions import poly_n

# Load calibration coefficients:
with open('rel_fluct_perc_calib_fit_coeffs.pickle', 'rb') as infile:
    fit_coeffs_dict = pickle.load(infile)
    fit_coeffs = list(reversed(fit_coeffs_dict.values()))
    
def rel_fluct_from_cutoff(cutoff):
    """Get the relative fluctuations from the post-selection fluctuation
    cutoff (both in %). Reproduces the fitting function used for the
    calibration."""
    return poly_n(cutoff, *fit_coeffs)

def cutoff_from_rel_fluct(rel_fluct):
    """Get the post-selection fluctuation cutoff that results in the given
    relative fluctuations for the retained shots (both in %)."""    
    a, b, c = fit_coeffs
    return (np.sqrt(b**2 + 4 * a * (rel_fluct - c)) - b) / (2 * a)
