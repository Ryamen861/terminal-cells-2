#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:26:56 2026

@author: ryanmoon
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import copy

def get_closest_node(G, node, pid=None, exclude_nodes=None):
    
    coords = node["coords"]
    
    if exclude_nodes is None:
        exclude_nodes = set()

    best = None  # tuple: (distance, node_id)

    for n, data in G.nodes(data=True):
        if n in exclude_nodes:
            continue

        d = np.linalg.norm(data["coords"] - coords)
        candidate = (d, n)

        if best is None or candidate < best:
            if pid == None:
                best = candidate
            else:
                print(G.nodes[n]["pid"])
                
                for i in range(5):
                    if G.nodes[n]["pid"] - n == G.nodes[node["id"]]["pid"]:
                        best = candidate
                
    return best[1]

def TFG_Dfly_ordered(filename):
    paths = {}
    node_count = 1

    # ---------- read file (path order preserved) ----------
    with open(filename) as file:
        for line in file:
            data = line.split()

            pid = int(data[-1])
            coords = np.array((float(data[2]),
                               float(data[3]),
                               float(data[4])))
            r = float(data[5])

            if pid not in paths:
                paths[pid] = []

            paths[pid].append({
                "id": node_count,
                "coords": coords,
                "radius": r,
                "path_id": pid,
            })

            node_count += 1

    # ---------- build directed tree ----------
    G = nx.DiGraph()

    for pid, nodes in paths.items():
        for n in nodes:
            G.add_node(n["id"], coords=n["coords"], radius=n["radius"], pid=n["path_id"])

    for pid, path in paths.items():
        for n in path:
            exclude = {m["id"] for m in path}
            parent = get_closest_node(G, n, pid=None, exclude_nodes=exclude)
            G.add_edge(parent, n["id"])
            
            if nx.is_directed_acyclic_graph(G):
                G.remove_edge(parent, n["id"])
                parent = get_closest_node(G, n, pid=pid, exclude_nodes=exclude)
                G.add_edge(parent, n["id"])


    # ---------- flip y-axis ----------
    coords = nx.get_node_attributes(G, 'coords')
    max_y = np.max([v[1] for v in coords.values()])
    coords = {k: (coords[k][0], max_y - coords[k][1], coords[k][2]) for k in coords}
    nx.set_node_attributes(G, coords, 'coords')

    # ---------- edge lengths ----------
    for u, v in G.edges():
        c1 = np.array(G.nodes[u]['coords'][:2])
        c2 = np.array(G.nodes[v]['coords'][:2])
        G[u][v]['length'] = np.linalg.norm(c1 - c2)

    # ---------- remove degree-2 nodes ----------
    cut_G = copy.deepcopy(G)

    changed = True
    while changed:
        print("Here")
        changed = False
        for n in list(cut_G.nodes()):
            if cut_G.in_degree(n) == 1 and cut_G.out_degree(n) == 1:
                parent = list(cut_G.predecessors(n))[0]
                child  = list(cut_G.successors(n))[0]

                new_len = cut_G[parent][n]['length'] + cut_G[n][child]['length']
                cut_G.add_edge(parent, child, length=new_len)
                cut_G.remove_node(n)

                changed = True
                break

    return G, cut_G

#%% Dragonfly testing

def basic_show_data(G):

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
             
        plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = 1)

    coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    xs = coords[:, 0]
    ys = coords[:, 1]

    xcent = 0.5*(np.max(xs) + np.min(xs))
    ycent = 0.5*(np.max(ys) + np.min(ys))

    lim = 250

    plt.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.show()
    
def plot_dfly(fname):    
    
    xs = []
    ys = []
    
    with open(fname) as file:
        for line in file:
            data = line.split()

            xs.append(float(data[2]))
            ys.append(float(data[3]))
    
            plt.scatter(xs, ys)
            plt.show()

from TMD_analysis import TMD, show_data
dflies = [154, 160, 171, 201]

# these files have this format: id,type,x,y,z,r,pid
# clarification: id is basically index, like in a csv
for dfly in dflies:
    
    file_name = f"dragonfly/C{dfly}/Source-Version/C{dfly}.swc"
    # file_name = f"dragonfly/C{dfly}/CNG version/C{dfly}.CNG.swc"
    plot_dfly(file_name)
    break
    
    # G, cut_G = TFG_Dfly_ordered(file_name)
    # basic_show_data(G)

        
    # TMD_coords = TMD(cut_G, min(list(cut_G.nodes())))
        
    # y axis is in increasing length of strand for persistence barcode
    # TMD_coords = sort_persistence_pairs(coords)
            
    # the upper left hand drawing will reflect L/R
    # show_data(G, cut_G, TMD_coords, dfly, "NA", "NA")
        


