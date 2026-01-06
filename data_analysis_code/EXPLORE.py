#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 16:09:18 2025

@author: ryanmoon
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


from convert_trace_to_network import trace_file_to_G

# following function copied over from test_tree.py in the network_simulation_code folder
def color_plot_walk(G, savename):

    fig, ax = plt.subplots(figsize=(2, 2))

    # maxLevel = max(nx.get_edge_attributes(G, 'level').values()) + 1
    maxLevel = 500

    print('max level:', maxLevel)

    palette = sns.dark_palette("red", maxLevel, reverse=True)

    for e in G.edges(data=True):
        # e is a tuple that looks like this:
        # (node_connected_by_edge, other_node_connected_by_edge, dict_of_attributes)
        # the dictionary holds level, length, and ange information
        
        # find the coordinates of the two nodes connected by this edge
        c0 = G.nodes[e[0]]['coords']
        c1 = G.nodes[e[1]]['coords']
        
        c = 'b'

        # # the level gives index for RGB value
        # c = palette[e[2]['level']]
        # if e[2]['level'] == 0:
        #     # if the level is zero, then make it blue
        #     # reminder: nodes of level zero are ones made in the initialization
        #      c = 'b'

        plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = .5)

    #plt.axis('equal')
    #plt.axis('off')

    coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    xs = coords[:, 0]
    ys = coords[:, 1]

    xcent = 0.5*(np.max(xs) + np.min(xs))
    ycent = 0.5*(np.max(ys) + np.min(ys))

    print(xcent, ycent)

    lim = 120

    plt.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.savefig(savename + '.pdf', bbox_inches='tight')
    plt.close()

side = "R"

for num in range(1, 27):
    filename = f"data/traces_L1/{num}_Tr9{side}.traces"
    G = trace_file_to_G(filename)
    
    color_plot_walk(G, "reproduction")
    break