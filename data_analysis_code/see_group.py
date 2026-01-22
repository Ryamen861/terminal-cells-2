#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 13:52:00 2026

@author: ryanmoon
"""

import matplotlib.pyplot as plt
import math

from TMD_analysis import draw_G, TFG

def plot_files(file_names, draw_func, simplified=False, max_cols=3, figsize=(12, 8)):
    """
    file_names : list of file paths
    draw_func  : function(file_name, ax) -> draws one plot on ax
    max_cols   : maximum number of subplot columns
    figsize    : figure size
    """
        
    n = len(file_names)
    if n == 0:
        return

    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # make axes iterable
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, fname in zip(axes, file_names):
        G, cut_G = TFG(fname)
        
        if simplified:
            draw_func(cut_G, ax)
        else:
            draw_func(G, ax)

    # turn off unused axes
    for ax in axes[len(file_names):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

done = False

flies_to_cue = []

while not done:
        
    new_entry = ""
    
    valid_request = False
    while not valid_request:
        
        fly_data = input("Which fly and side do you want? (fly side):\n").strip()

        if fly_data == 'quit':
            done = True
            break
        
        fly, side = tuple(fly_data.split(" "))
        
        try:
            fly = int(fly)
        
        except ValueError:
            break
        
        
        if (0 < fly and fly < 29) and (side == "L" or side == "R"):
            new_entry = str(fly) + "_Tr9" + side
            flies_to_cue.append(new_entry)
            
        else:
            print("That is not a valid fly, flies range from 1-28")
            break

flies_to_cue = [f"data/traces_L3/{i}.traces" for i in flies_to_cue]
plot_files(flies_to_cue, draw_G)
        
    

