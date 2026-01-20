#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:26:56 2026

@author: ryanmoon
"""
import xml.etree.ElementTree as ET
import gzip
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy


dflies = [154, 160, 171, 201]


def get_closest_node(G, coords, exclude_nodes=None):
    if exclude_nodes is None:
        exclude_nodes = set()

    best = None  # tuple: (distance, node_id)

    for n, data in G.nodes(data=True):
        if n in exclude_nodes:
            continue

        d = np.linalg.norm(data["coords"] - coords)
        candidate = (d, n)

        if best is None or candidate < best:
            best = candidate

    return best[1]

def TFG_Dfly(filename):
    '''Takes in a .trace file, returns a DiGraph object constructed from trace
    TFG is short for Trace File to Graph'''
    
    paths = {}
    
    node_count = 1
        
    with open(filename) as file:
        lines = file.readlines()
        # id,type,x,y,z,r,pid

        
        for line in lines:
            data = line.split(" ")            
            pid = int(data[-1].strip())
            
            paths[pid] = []
    
        for line in lines:
            data = line.split(" ")            
            coords = np.array((float(data[2]), float(data[3]), float(data[4])))
            pid = int(data[-1].strip())
                        
            new_node = {
                "id": node_count,
                "coords": coords,
                "radius": np.linalg.norm(coords), # why is it all '1' in the file?
                "path_id": pid,
                "is_first": False
                }
        
            # update the dictionary to have the new node added to the corresponding path
            previous_list = paths[pid] # fetch the list
            if len(previous_list) == 0:
                new_node["is_first"] = True
            previous_list.append(new_node) # update the list
            paths[pid] = previous_list # save the list
            
            node_count += 1
            
    G = nx.DiGraph()
    
    for pid, nodes in paths.items():
        for n in nodes:
            G.add_node(n["id"], coords=n["coords"], radius=n["radius"])
    
    last_node_in_path = {}
    
    for path_id, path in paths.items():
        for n in path:

            if path_id == -1:
                continue
        
            if not n["is_first"]:
                # extension
                parent = last_node_in_path[pid]
                G.add_edge(parent, n["id"])
            else:
                # branch
                exclude = {m["id"] for m in nodes if m["path_id"] == n["path_id"]}
                parent = get_closest_node(G, n["coords"], exclude_nodes=exclude)
                G.add_edge(parent, n["id"])
            
            last_node_in_path[pid] = n["id"]

    
    coords = nx.get_node_attributes(G, 'coords')
    # flip y-axis for proper AP orientation
    max_y = np.max([v[1] for v in coords.values()])
    coords = {k: (coords[k][0], max_y - coords[k][1], coords[k][2]) for k in coords.keys()}
    
    nx.set_node_attributes(G, coords, 'coords')
    
    for e in G.edges():
        c1 = np.array(G.nodes[e[0]]['coords'][:2])
        c2 = np.array(G.nodes[e[1]]['coords'][:2])
        G[e[0]][e[1]]['length'] = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    
    # cut out the degree 2 nodes
    cut_G = copy.deepcopy(G)
    more_to_cut = True
    while more_to_cut:
        more_to_cut = False
        # if there are no more degree 2 nodes, more_to_cut will stay False and exit loop
        for n in list(cut_G.nodes()):
            if cut_G.in_degree(n) == 1 and cut_G.out_degree(n) == 1:
                parent = list(cut_G.predecessors(n))[0]
                child  = list(cut_G.successors(n))[0]
    
                # accumulate length
                new_len = cut_G[parent][n]['length'] + cut_G[n][child]['length']
    
                # reconnect
                cut_G.add_edge(parent, child, length=new_len)
    
                # remove middle node
                cut_G.remove_node(n)
                more_to_cut = True
                break

    return G, cut_G
    
    

        

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
                "is_first": False
            })

            node_count += 1

    # mark first node of each path
    for pid, nodes in paths.items():
        nodes[0]["is_first"] = True

    # ---------- build directed tree ----------
    G = nx.DiGraph()

    for pid, nodes in paths.items():
        for n in nodes:
            G.add_node(n["id"], coords=n["coords"], radius=n["radius"])

    last_node_in_path = {}

    for pid, path in paths.items():
        for n in path:

            if not n["is_first"]:
                parent = last_node_in_path[pid]
                G.add_edge(parent, n["id"])

            else:
                exclude = {m["id"] for m in path}
                parent = get_closest_node(G, n["coords"], exclude_nodes=exclude)
                G.add_edge(parent, n["id"])

            last_node_in_path[pid] = n["id"]

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

    print(xcent, ycent)

    lim = 120

    plt.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('on')

    plt.show()


# these files have this format: id,type,x,y,z,r,pid
# clarification: id is basically index, like in a csv
for dfly in dflies:
    
    file_name = f"dragonfly/C{dfly}.CNG.swc"
    G, cut_G = TFG_Dfly_ordered(file_name)
    # coords = TMD(cut_G, min(list(cut_G.nodes())))
            
    # y axis is in increasing length of strand for persistence barcode
    # TMD_coords = sort_persistence_pairs(coords)
            
    # the upper left hand drawing will reflect L/R
    # show_data(G, cut_G, TMD_coords, dfly, "")
    
    basic_show_data()
