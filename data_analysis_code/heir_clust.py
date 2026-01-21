#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 12:39:52 2026

@author: ryanmoon
"""

from itertools import combinations
import numpy as np
import networkx as nx
from TMD_analysis import TMD_flies


def make_hist(diagram, size):
    '''Takes in a diagram, which is just a set of TMD coords, and returns a quasi-histogram.
    It will actually be just a np.array
    
    
    This histogram represents the number of bars alive at a given point (looking at the barcode diagram)'''
    
    new_hist = np.array([])
    
    x = 1 # this is the x value (this is what is meant by at a given point)
    resolution = 0.5
    
    while x < size:
        # define a new frequency
        new_frequency = 0

        # add to the frequency every time we find a bar alive at this x value
        for birth, death in diagram:
            
            if birth < x and x < death:
                new_frequency += 1
                
        # add that to the dictionary
        new_hist = np.append(new_hist, new_frequency)
                
        x += resolution
        
    # however, it is not guaranteed that the sizes/lengths of the TMD coords are the same
    # for every TMD diagram, so we will have to make it up by adding in empty frequencies
    # for the shorter one
    curr_len = len(new_hist)
    if curr_len < size:
        how_many_more = size - curr_len
        
        for _ in range(how_many_more):
            new_hist = np.append(new_hist, new_frequency)
            x += resolution
    
    return new_hist
    

def find_dbar(D1, D2):
    '''Apparently, this method "fails to capture the local differences between the branching
    structures of similar neuronal trees'''
    
    max_len = max([j for i, j in D1] + [i for i, j in D1] + [j for i, j in D2] + [i for i, j in D2])
        
    D1_hist = make_hist(D1, max_len)
    D2_hist = make_hist(D2, max_len)
        
    return np.sum(abs(D1_hist - D2_hist))


def metric_func(D1, D2):
    return np.sum(abs(D1 - D2))

# unsure about following implementation of bottleneck distance
def point_dist(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))

def diag_dist(p):
    return abs(p[1] - p[0]) / 2

def find_db(D1, D2):
    D1 = list(D1)
    D2 = list(D2)

    n1, n2 = len(D1), len(D2)
    n = max(n1, n2)

    # --- candidate epsilons ---
    eps_vals = set()

    for p in D1:
        eps_vals.add(diag_dist(p))
        for q in D2:
            eps_vals.add(point_dist(p, q))

    for q in D2:
        eps_vals.add(diag_dist(q))

    eps_vals = sorted(eps_vals)

    # --- epsilon decision via bipartite matching ---
    def can_match(eps):
        G = nx.Graph()

        left_nodes  = [("L", i) for i in range(n)]
        right_nodes = [("R", j) for j in range(n)]

        G.add_nodes_from(left_nodes,  bipartite=0)
        G.add_nodes_from(right_nodes, bipartite=1)

        for i in range(n):
            for j in range(n):

                # real–real
                if i < n1 and j < n2:
                    if point_dist(D1[i], D2[j]) <= eps:
                        G.add_edge(("L", i), ("R", j))

                # real–diagonal
                elif i < n1 and j >= n2:
                    if diag_dist(D1[i]) <= eps:
                        G.add_edge(("L", i), ("R", j))

                # diagonal–real
                elif i >= n1 and j < n2:
                    if diag_dist(D2[j]) <= eps:
                        G.add_edge(("L", i), ("R", j))

                # diagonal–diagonal (always free)
                else:
                    G.add_edge(("L", i), ("R", j))

        matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=left_nodes)
        return len(matching) // 2 == n

    # --- binary search over eps ---
    lo, hi = 0, len(eps_vals) - 1
    ans = eps_vals[hi]

    while lo <= hi:
        mid = (lo + hi) // 2
        eps = eps_vals[mid]

        if can_match(eps):
            ans = eps
            hi = mid - 1
        else:
            lo = mid + 1

    return ans






barcodes = TMD_flies()
# barcodes is a list of dict(file_name: TMD_coords)

distances = {}

k = nx.Graph()

combos = combinations(barcodes, 2)

for combo in combos:
    # grab a random combo, take the two diagrams out of it
    D1_dict, D2_dict = combo
    
    # get the fly name and actual coords/diagram (D1 or D2) out of it
    fly1_name, D1 = list(D1_dict.items())[0]
    fly2_name, D2 = list(D2_dict.items())[0]

    # compute the metric
    new_dbar = find_dbar(D1, D2)
    
    print(fly1_name, fly2_name, new_dbar)
    
    # embed metric in graph
    k.add_nodes_from([fly1_name, fly2_name])
    k.add_edge(fly1_name, fly2_name, distance=new_dbar)
    
    
goal_num_edges = len(list(k.nodes())) - 1
curr_num_edges = len(list(k.edges()))
    
num_to_cut = curr_num_edges - goal_num_edges

edges = sorted(list(k.edges(data=True)), key=lambda x: x[-1]["distance"], reverse=True)

for _ in range(num_to_cut):
    removing_edge = edges[0]
    edges.pop(0)
    
    k.remove_edge(removing_edge[0], removing_edge[1])
    
# can manually check
# print(k.nodes)
# print(k.edges(data=True))
    

# we should now be left with one number line that contains all fly names (L and R) as nodes
# and their distances as weights on the edges
# now, we are ready to perform heirarchical clustering


#%%
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

barcodes = TMD_flies()
# barcodes is a list of dict(file_name: TMD_coords)

X = [list(list(barcode.values())[0]) for barcode in barcodes]

for i in X:
    print(i)
    print("\n")

Z = linkage(X, metric=find_dbar)

dn = dendrogram(Z)
plt.show()



