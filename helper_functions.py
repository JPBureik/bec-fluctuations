#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 08:52:22 2022

@author: jp
"""

# Standard library imports:
import multiprocessing as mp
import numpy as np
import progressbar

def poly_n(x, *coeffs):
    """Polynomial of order len(coeffs) - 1. Coefficients in decreasing order.
    Ex: Quadratic: poly_n(x, *[a, b, c]) returns a * x**2 + b * x + c."""
    return sum(coeff * x**order
               for order, coeff in enumerate(reversed(coeffs)))

def multiproc_list(input_list, func, keep_order=True, show_pbar=False,
                   desc=None):
    """Multiprocessing for lists and numpy arrays.

    Returns [func(element) for element in input_list] (ordered or unordered).
    The function func can only accept the individual list or array elements.
    The list or array cannot be a class or instance attribute. For long lists
    performance may benefit from not keeping the index order. Optionally show
    progress bar with its description."""
    # Keep track of index:
    if keep_order:
        def pfunc(idx_val_func_tuple):
            idx, val, func = idx_val_func_tuple
            return idx, func(val)
    else:
        pfunc = func
    input_array = (input_list if isinstance(input_list, np.ndarray)
                              else np.array(input_list))
    # Split into number of elements:
    split = np.array_split(input_array, len(input_list))
    # Set up progress bar:
    if show_pbar:
        nb_of_tasks = len(input_list)
        completed = 0
        pbar_desc = desc if desc else 'Processing'
        widgets = [
            pbar_desc + ': ', progressbar.Percentage(),
            ' (' , progressbar.SimpleProgress(), ')',
            ' ',  progressbar.GranularBar(
                markers=" ░▒▓█",
                left='|',
                right='|'
                ),
            ]
        pbar = progressbar.ProgressBar(
            widgets=widgets,
            max_value=nb_of_tasks,
            redirect_stdout=True,
            suffix=' {elapsed}'
            ).start()
    # Create workers:
    queue = mp.Queue()
    workers = list()
    results = {i: None for i in range(len(split))} if keep_order else []
    for idx, element in enumerate(split):
        pfunc_args = (idx, element[0], func) if keep_order else element[0]
        workers.append(
            mp.Process(
                target=lambda x: queue.put(pfunc(x)),
                args=(pfunc_args,)
                )
            )
    # Start up workers:
    for p in workers:
        p.start()
    # Harvest results:
    for p in workers:
        if keep_order:
            result = queue.get()
            results[result[0]] = result[1]
        else:
            results.append(queue.get())
        # Update progress bar:
        if show_pbar:
            completed += 1
            pbar.update(completed, force=True)
    if show_pbar:
        pbar.finish()
    # Terminate workers:
    for p in workers:
        p.join()
    return ([results[idx] for idx in sorted(results.keys())] if keep_order
                                                             else results)
