#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:55:47 2022

@author: jp
"""

import socket
from os import listdir
from os.path import isfile, join
import pandas as pd
from matplotlib import pyplot as plt
import boost_histogram as bh
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
import pickle



save_fig = False
DET_EFF = 0.53
# n_factors = (0.5, 0.7, 1.0, 1.3, 1.7)
# n_factors = (0.7, 1.0, 1.3, 1.7)
n_factors = (1.0, 1.3)

# Resolve hostname to get data directory:
hostname = socket.gethostname()
if hostname == 'catarella':
    # data_basepath = '/home/jp/Documents/data/recentered_data/indiv_mean/rc_'
    data_basepath = '/home/jp/Documents/data/recentered_data/indiv_uj_mean/rc_'
    # data_basepath = '/home/jp/Documents/data/recentered_data/same_global_mean/rc_'
    # data_basepath = '/home/jp/Documents/prog/work/bec-fluctuations/data/mat_'
    # data_basepath = {n_factor: f"/home/jp/Documents/data/recentered_data/indiv_var_n_without_uj_24/{str(n_factor).replace('.', 'p')}/rc_" for n_factor in n_factors}
    # data_basepath = {n_factor: f"/home/jp/Documents/data/recentered_data/indiv_var_n_without_uj_22/{str(n_factor).replace('.', 'p')}/rc_" for n_factor in n_factors}
    data_basepath = {n_factor: f"/home/jp/Documents/data/recentered_data/indiv_var_n/{str(n_factor).replace('.', 'p')}/rc_" for n_factor in n_factors}
    # fluct_perc_vals = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 7.5, 10, 30]
    # fluct_perc_vals = [0.1, 0.25, 0.5, 1, 2]#, 10, 30]
    fluct_perc_vals = [2, 5, 30]
    with open('/home/jp/Documents/prog/work/bec-fluctuations/lattice_atom_number_calibration_data.pickle', 'rb') as infile:
            lattice_atom_number_calibration = pickle.load(infile)
elif hostname == 'zito':
    data_basepath = '/home/helium/Documents/recentered_data/indiv_uj_mean/rc_'
    # data_basepath = '/home/helium/Documents/recentered_data/same_global_mean/rc_'
    fluct_perc_vals = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 7.5, 10, 30]
elif hostname == 'jp-MS-7C02':
    # data_basepath = '/home/jp/Documents/data/recentered_data/indiv_uj_mean/rc_'
    # data_basepath = '/home/jp/Documents/data/recentered_data/indiv_var_n/1p0/rc_'
    data_basepath = {n_factor: f"/home/jp/Documents/data/recentered_data/indiv_var_n/{str(n_factor).replace('.', 'p')}/rc_" for n_factor in n_factors}
    fluct_perc_vals = [2, 5, 30]
    with open('/home/jp/Documents/prog/work/bec-fluctuations/lattice_atom_number_calibration_data.pickle', 'rb') as infile:
            lattice_atom_number_calibration = pickle.load(infile)
            
def calibrate_lattice_atom_number(uj):
    return sum(val * uj ** key for key, val in lattice_atom_number_calibration.items())

def eta(uj):
    return calibrate_lattice_atom_number(uj)/5000

# variance = dict.fromkeys(fluct_perc_vals)
# relative_fluctuations = dict.fromkeys(fluct_perc_vals)


mcp_data = dict.fromkeys(n_factors)
variance = dict.fromkeys(n_factors)
relative_fluctuations = dict.fromkeys(n_factors)


for n_factor in n_factors:
    bec_peak_atom_numbers = dict.fromkeys(fluct_perc_vals)
    shot_atom_numbers = dict.fromkeys(fluct_perc_vals)
    shot_to_shot_fluctuations = dict.fromkeys(fluct_perc_vals)
    variance[n_factor] = dict.fromkeys(fluct_perc_vals)
    relative_fluctuations[n_factor] = dict.fromkeys(fluct_perc_vals)

    # Main loop:
    for fluct_perc in tqdm(fluct_perc_vals):

        data_path = data_basepath[n_factor] + str(fluct_perc).replace('.', 'p') + 'pct'
        # data_path = data_basepath + str(fluct_perc).replace('.', 'p') + 'pct'
        k_max_base = 1.72e-3  # in units of k_d

        k_min = 0
        k_max = 0.15

        data_files = [join(data_path, f) for f in listdir(data_path)]

        mcp_data[n_factor] = dict()

        # Load recentered data:
        for file in data_files:
            lattice_depth_str = file.split('.h5')[0].split('/')[-1].split('_')[0].split('uj')[-1]
            if 'p' in lattice_depth_str:
                lattice_depth_str = lattice_depth_str.replace('p', '.')  # Decimal sep
            lattice_depth = str(float(lattice_depth_str))  # Strip leading zeros
            fluct_perc = file.split('.h5')[0].split('/')[-1].split('_')[1].split('pct')[0]
            if 'p' in fluct_perc:
                fluct_perc = float(fluct_perc.replace('p', '.'))
            else:
                fluct_perc = float(fluct_perc)
            if lattice_depth not in ('7.5'):
                if file.split('.')[-1] == 'h5':
                    mcp_data[n_factor][lattice_depth] = pd.read_hdf(file)
                elif file.split('.')[-1] == 'mat':
                    matlab_data_all = loadmat(file)['momentum_lattice_axis']
                    matlab_data = []
                    mcp_data[n_factor][lattice_depth] = pd.DataFrame(columns=['k_m45', 'k_h', 'k_p45'])
                    for idx, line in enumerate(matlab_data_all):
                        try:
                            np.count_nonzero(line)
                        except ValueError:
                            k_h_matlab = line[1]
                            k_m45_matlab = line[0]
                            k_p45_matlab = line[2]
                            k_h = [j[0] for j in k_h_matlab]
                            k_m45 = [j[0] for j in k_m45_matlab]
                            k_p45 = [j[0] for j in k_p45_matlab]
                            mcp_data[n_factor][lattice_depth].at[idx] = pd.Series(dtype=object)
                            mcp_data[n_factor][lattice_depth]['k_h'].at[idx] = np.array(k_h)
                            mcp_data[n_factor][lattice_depth]['k_m45'].at[idx] = np.array(k_m45)
                            mcp_data[n_factor][lattice_depth]['k_p45'].at[idx] = np.array(k_p45)

        bec_peak_atom_numbers[fluct_perc] = dict.fromkeys(mcp_data[n_factor])
        shot_atom_numbers[fluct_perc] = dict.fromkeys(mcp_data[n_factor])
        variance[n_factor][fluct_perc] = dict.fromkeys(mcp_data[n_factor])
        relative_fluctuations[n_factor][fluct_perc] = dict.fromkeys(mcp_data[n_factor])

        # Count number of atoms in BEC and in entire shot:
        for dataset in mcp_data[n_factor]:
            bec_peak_atom_numbers[fluct_perc][dataset] = dict.fromkeys(mcp_data[n_factor][dataset].index.dropna())
            shot_atom_numbers[fluct_perc][dataset] = dict.fromkeys(mcp_data[n_factor][dataset].index.dropna())
            for run in mcp_data[n_factor][dataset]['k_h'].index.dropna():

                indices = dict.fromkeys(['k_h', 'k_m45', 'k_p45'])

                for axis in indices:
                    indices[axis] = (k_min < abs(mcp_data[n_factor][dataset][axis][run])) * (abs(mcp_data[n_factor][dataset][axis][run]) < k_max)
                indices_all = np.where(indices['k_h'] * indices['k_m45'] * indices['k_p45'])
                assert len(list(mcp_data[n_factor][dataset]['k_h'][run][indices_all])) == len(list(mcp_data[n_factor][dataset]['k_m45'][run][indices_all])) == len(list(mcp_data[n_factor][dataset]['k_p45'][run][indices_all]))
                bec_peak_atom_numbers[fluct_perc][dataset][run] = len(list(mcp_data[n_factor][dataset]['k_h'][run][indices_all]))
                assert len(list(mcp_data[n_factor][dataset]['k_h'][run])) == len(list(mcp_data[n_factor][dataset]['k_m45'][run])) == len(list(mcp_data[n_factor][dataset]['k_p45'][run]))
                shot_atom_numbers[fluct_perc][dataset][run] = len(list(mcp_data[n_factor][dataset]['k_h'][run]))


        # Calculate variance and normalize:
        for dataset in mcp_data[n_factor]:
            variance[n_factor][fluct_perc][dataset] = np.var(list(bec_peak_atom_numbers[fluct_perc][dataset].values()))
            relative_fluctuations[n_factor][fluct_perc][dataset] = variance[n_factor][fluct_perc][dataset]/np.mean(list(shot_atom_numbers[fluct_perc][dataset].values())) / eta(float(dataset))


#%% Plot variance

with open('/home/jp/Documents/prog/work/bec-fluctuations/lattice_atom_number_calibration_data.pickle', 'rb') as infile:
    lattice_atom_number_calibration = pickle.load(infile)

def calibrate_lattice_atom_number(uj):
    return sum(val * uj ** key for key, val in lattice_atom_number_calibration.items())

fig, ax1 = plt.subplots(figsize=(16, 9))
ax2 = ax1.twinx()

def make_plot_colors(color_str):
    return [tuple([int(i)/255 for i in color_str.split('\n')[j].split(',')])
            for j in range(len(color_str.split('\n')))]

# plot_color_str = """228,26,28
# 55,126,184
# 77,175,74
# 152,78,163
# 255,127,0"""

plot_color_str = """228,26,28
55,126,184
77,175,74
152,78,163
255,127,0"""


lattice_depth_plt = sorted(float(key) for key in shot_atom_numbers[fluct_perc].keys())

plot_colors = make_plot_colors(plot_color_str)
plot_colors = dict(zip(fluct_perc_vals, plot_colors))

variance_plot = dict.fromkeys(n_factors)


linestyle_list = [
    'solid',
    'dotted',
    'dashed',
    'dashdot'
    ]

linestyle = dict(zip(n_factors,linestyle_list))


for n_factor in n_factors:
    for fluct_perc in fluct_perc_vals:

        variance_plot[n_factor] = [variance[n_factor][fluct_perc][str(key)] for key in lattice_depth_plt]
        relative_fluctuations_plot = [relative_fluctuations[n_factor][fluct_perc][str(key)] for key in lattice_depth_plt]
        bec_peak_atom_numbers_plot = [np.mean(list(bec_peak_atom_numbers[fluct_perc][str(key)].values())) for key in lattice_depth_plt]
        atom_numbers_plot = [n_factor*np.mean(list(shot_atom_numbers[fluct_perc][str(key)].values())) for key in lattice_depth_plt]
        # Plot fluctuations:
        real_atom_nb_k = np.mean(np.mean([list(shot_atom_numbers[fluct_perc][str(key)].values()) for key in lattice_depth_plt]))/np.mean([eta(float(key)) for key in lattice_depth_plt])/1000
        fluct_std_perc = 100*np.mean([np.std(list(shot_atom_numbers[fluct_perc][str(key)].values()))/np.mean(list(shot_atom_numbers[fluct_perc][str(key)].values())) for key in lattice_depth_plt for key in lattice_depth_plt])
        print(f'{fluct_perc} - {fluct_std_perc}%')
        # file_nb =
        plot_label_var = f'{real_atom_nb_k:.1f}k atoms: variance: PS-Lim: {fluct_perc}% -> rel. fluct: {fluct_std_perc:.2f}%'
        plot_label_bec = f'{real_atom_nb_k:.1f}k atoms: Mean BEC Atom Number: PS-Lim: {fluct_perc}% -> rel. fluct: {fluct_std_perc:.2f}%'
        plot_label_rf = f'{real_atom_nb_k:.1f}k atoms: Relative fluct: PS-Lim: {fluct_perc}% -> rel. fluct: {fluct_std_perc:.2f}%'
        plot_label_sf = f'{real_atom_nb_k:.1f}k atoms: Shot noise fluct: PS-Lim: {fluct_perc}% -> rel. fluct: {fluct_std_perc:.2f}%'
        plot_label_an = f'{real_atom_nb_k:.1f}k atoms: Mean Total Atom Number: PS-Lim: {fluct_perc}% -> rel. fluct: {fluct_std_perc:.2f}%'
        # ax1.plot(lattice_depth_plt, variance_plotrelative_fluctuations, color=plot_colors[n_factor], label=plot_label_var)#, marker='*')
        # ax2.scatter(lattice_depth_plt, bec_peak_atom_numbers_plot, color=plot_colors[fluct_perc], label=plot_label_bec, marker='+')
        # ax2.scatter(lattice_depth_plt, atom_numbers_plot, color=plot_colors[fluct_perc], label=plot_label_an)
        ax1.plot(lattice_depth_plt, relative_fluctuations_plot, color=plot_colors[fluct_perc], linestyle=linestyle[n_factor], label=plot_label_rf)
    # ax2.plot(
    #     lattice_depth_plt,
    #     [calibrate_lattice_atom_number(uj)*n_factor for uj in lattice_depth_plt],
    #     color=plot_colors[n_factor],
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
# plt.title('variance[n_factor] of the atom number in the BEC for ' + r'$k_{max}$' + ' = ' + f"{k_max} " + r'$k_d$')
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='right', bbox_to_anchor=(0.5, 0.1, 0.4, 0.8))
plt.tight_layout()
plt.show()

if save_fig:
    fig.savefig('/home/jp/Pictures/prog/moments/rel_fluct_0p15_kd_with_global_mean_atom_number.png', dpi=fig.dpi)


#%%

# def s(uj):
#     return -4.798 + 8.918 * uj ** (1 / 4.58) - (1.739*10**(-2))*uj + (3.148*10**(-5))*uj**2


# s_list = []
# for i in range(100):
#     s_list.append(s(i))

# # plt.plot(s_list)

# mult_factor = 0.5
# add_factor = -6.7

# amp = {i*5000: relative_fluctuations[i][fluct_perc_vals[0]]['20.0'] for i in relative_fluctuations}

# plt.figure()
# plt.scatter(
#     amp.keys(),
#     amp.values(),
#     color='b',
#     label='Rel. Fluct. Ampl.'
#     )
# plt.plot(
#     amp.keys(),
#     [mult_factor * i ** (1/3) + add_factor for i in amp.keys()],
#     color='r',
#     label=str(mult_factor) + r'$ \times N^{\frac{1}{3}}$ + ' + str(add_factor)
#     )
# plt.legend()
# plt.show()