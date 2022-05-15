#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:41:51 2022

@author: jp
"""

# Standard library imports:
import socket
import os

# Local imports:
from mcpmeas.mcp_process import McpProcess

ADD_NEW = None#'07p5'
EXCLUDE = 'uj01'

# Setup: Set file path to MCP data directory:

hostname = socket.gethostname()

if hostname == 'catarella':
    data_basepath = '/home/jp/Documents/data/mcp_data/5k'
    save_path = '/home/jp/Documents/data/recentered_data/indiv_all/'

elif hostname == 'zito':
    data_basepath = '/home/helium/Documents/mcp_data/5k'
    save_path = '/home/helium/Documents/recentered_data/indiv_all/'

elif hostname == 'jp-MS-7C02':
    data_basepath = '/home/jp/Documents/data/mcp_data/5k'
    save_path = '/home/jp/Documents/data/recentered_data/indiv_all/'

datapaths = [x[0] for x in os.walk(data_basepath)][1:]

if os.path.join(data_basepath, EXCLUDE) in datapaths:

    datapaths.remove(os.path.join(data_basepath, EXCLUDE))

# Add single dataset:
# if ADD_NEW:
#     for dataset in datapaths:
#         if str(ADD_NEW) in dataset:
#             datapaths = [dataset]

#%% Recenter:


for dataset in datapaths:

    # Load TDC data files:
    uj_str = dataset.split('/')[-1].split('uj')[-1]
    if 'p' in uj_str:
        uj_val = float(uj_str.replace('p', '.'))
    else:
        uj_val = float(uj_str)
    mcp_process = McpProcess(dataset)
    mcp_process.load_reconstr_data()
    # Recenter:
    (feat_sel, angle_corr) = ('all', 'sts') if uj_val <= 20 else ('center', 'off')
    mcp_process.recenter_data(
        lattice_depth=uj_val,
        fluct_perc=1000,
        feat_sel=feat_sel,
        angle_corr=angle_corr,
        )
    uj_str = dataset.split('/')[-1] + '_'
    save_filepath = save_path + uj_str + 'all.mat'
    # Save:
    mcp_process.save_data(save_filepath)



