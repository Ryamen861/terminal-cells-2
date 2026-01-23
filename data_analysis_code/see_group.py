#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 13:52:00 2026

@author: ryanmoon
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import networkx as nx
import seaborn as sns


from TMD_analysis import TFG

def get_G_props(G):
    leaves = [i for i in G.nodes() if G.degree(i) == 1]
    
    xs = []
    ys = []
    zs = []
    
    for coord in nx.get_node_attributes(G, "coords").values():
        xs.append(coord[0])
        ys.append(coord[1])
        zs.append(coord[2])

    avg_coord = []
    
    for ns in [xs, ys, zs]:
        avg_coord.append(round(float(np.mean(np.array(ns)))))
    
    return f"{len(leaves)}, ({avg_coord[0]}, {avg_coord[1]}, {avg_coord[2]})"

def draw_G(G, ax, side):
    maxLevel = max(G.nodes()) + 1
    palette = sns.dark_palette("red", maxLevel, reverse=True)
    
    for e in G.edges(data=True):
        # e is a tuple that looks like this:
        # (node_connected_by_edge, other_node_connected_by_edge, dict_of_attributes)
        # the dictionary holds level, length, and ange information
        
        # find the coordinates of the two nodes connected by this edge
        c0 = G.nodes[e[0]]['coords']
        c1 = G.nodes[e[1]]['coords']
        
        # the level gives index for RGB value
        c = palette[e[0]]
        if e[0] == min(list(G.nodes())):
            c = 'b'
            
        if e[1] == max(list(G.nodes())):
            c = 'g'
             
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = 0.5) # adjust thickness
            
    trace_coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    xs = trace_coords[:, 0]
    ys = trace_coords[:, 1]

    xcent = 0.5*(np.max(xs) + np.min(xs))
    ycent = 0.5*(np.max(ys) + np.min(ys))

    lim = 300

    ax.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    ax.text(0, 0, get_G_props(G), fontsize=7, ha="center", va="center", color="black")
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    if side == "L":
        ax.invert_xaxis()
    
    ax.axis('off')

def plot_files(file_names, draw_func, simplified=False, max_cols=4):
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

    fig, axes = plt.subplots(rows, cols)

    # make axes iterable
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, fname in zip(axes, file_names):
        G, cut_G = TFG(f"data/traces_L3/{fname}.traces")
        
        if simplified:
            draw_func(cut_G, ax, fname[-1])
        else:
            draw_func(G, ax, fname[-1])
            
        ax.set_title(f"{fname}")

    # turn off unused axes
    for ax in axes[len(file_names):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()




############################################
complete_Pclusters = [
    [
     "28 R",
     "22 L",
     "12 L",
     "26 L",
     "6 R",
     "2 L",
     "17 R",
     "2 R",
     "12 R",
     "22 R",
     "21 R",
     "4 L"
     ],
    
    [
     "28 L",
     "16 L",
     "16 R",
     "23 L",
     "1 L",
     "20 R",
     "18 L",
     "26 R",
     "14 L",
     "11 R",
     "15 L",
     "25 L"
     ],
    
    [
     "25 R",
     "11 L",
     "7 R",
     "10 R",
     "8 L",
     "20 L",
     "15 R",
     "14 R",
     "10 L",
     "23 R"
     ]
    
    ]


done = False

flies_to_cue = []

# while not done:
        
#     new_entry = ""
    
#     valid_request = False
#     while not valid_request:
        
#         fly_data = input("Which fly and side do you want? (fly side):\n").strip().upper()

#         if fly_data == 'QUIT':
#             done = True
#             break
        
#         fly, side = tuple(fly_data.split(" "))
        
#         try:
#             fly = int(fly)
        
#         except ValueError:
#             break
        
        
#         if (0 < fly and fly < 29) and (side == "L" or side == "R"):
#             new_entry = str(fly) + "_Tr9" + side
#             flies_to_cue.append(new_entry)
            
#         else:
#             print("That is not a valid fly, flies range from 1-28 and sides are either L or R")
#             break

# plot_files(flies_to_cue, draw_G, simplified=True)
        
for i, Pcluster in enumerate(complete_Pclusters):
    cluster = []
    
    for data in Pcluster:
        fly, side = tuple(data.split(" "))
        new_entry = str(fly) + "_Tr9" + side
        cluster.append(new_entry)
        
    plot_files(cluster, draw_G, simplified=True)
    plot_files(cluster, draw_G, simplified=False)
    

    
    


