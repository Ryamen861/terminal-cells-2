#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:38:24 2026

@author: ryanmoon
"""

import xml.etree.ElementTree as ET
import gzip
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from convert_trace_to_network import trace_file_to_G

def color_plot_walk(G, savename):

    fig, ax = plt.subplots(figsize=(2, 2))

    # maxLevel = max(nx.get_edge_attributes(G, 'level').values()) + 1

    # print('max level:', maxLevel)

    # palette = sns.dark_palette("red", maxLevel, reverse=True)

    for e in G.edges(data=True):
        # e is a tuple that looks like this:
        # (node_connected_by_edge, other_node_connected_by_edge, dict_of_attributes)
        # the dictionary holds level, length, and ange information
        
        # find the coordinates of the two nodes connected by this edge
        c0 = G.nodes[e[0]]['coords']
        c1 = G.nodes[e[1]]['coords']

        # the level gives index for RGB value
        c = 'b'

        plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = .5)

    #plt.axis('equal')
    #plt.axis('off')

    coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    xs = coords[:, 0]
    ys = coords[:, 1]

    xcent = 0.5*(np.max(xs) + np.min(xs))
    ycent = 0.5*(np.max(ys) + np.min(ys))

    lim = 300

    plt.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    plt.savefig(savename + '.pdf', bbox_inches='tight')
    plt.close()
    
def get_closest_node(G, n_coords):

    all_pts = nx.get_node_attributes(G, 'coords')

    dist_dict = {k: np.sqrt(np.sum((all_pts[k] - n_coords)**2)) for k in all_pts.keys()}

    sorted_by_dist = dict(sorted(dist_dict.items(), key=lambda item: item[1]))
    #closest = sorted_by_dist[0]

    #print(list(sorted_by_dist.keys())[:5])
    return list(sorted_by_dist.keys())[0]
    
def trace_file_to_G(filename):

    input = gzip.open(filename, 'r')
    tree = ET.parse(input)
    root = tree.getroot()
    
    special_xs = []
    special_ys = []
    special_zs = []

    G = nx.Graph()
    node_count = 1

    for path in root:

        if path.tag == 'path':
            # print(path.tag, path.attrib.keys())

            pathDict = path.attrib
            
            if pathDict["usefitted"] == 'false':
            # ['id', 'swctype', 'color', 'channel', 'frame', 'spines', 'usefitted', 'fitted',
            #     'startson', 'startx', 'starty',
            #     'startz', 'startsindex', 'name', 'reallength'])

                # print('now on path id', pathDict[path_dict_keys[0]])

                node_list = []

                first_pt = True

                for n in path.iter(tag='point'):

                    nDict = n.attrib
                    keys = list(nDict.keys())

                    # print(nDict)

                    xd = np.round(float(nDict[keys[3]]), 4)
                    yd = np.round(float(nDict[keys[4]]), 4)
                    zd = np.round(float(nDict[keys[5]]), 4)
                    r = float(nDict[keys[0]])

                    # if first node and node not 1, add edge to the closest node in existing network

                    if first_pt and node_count > 1:
                        special_xs.append(xd)
                        special_ys.append(yd)
                        special_zs.append(zd)
                        first_pt = False

                        #print('adding edge', node_count, get_closest_node(G, np.array((xd, yd, zd))))
                        neighbor_node = get_closest_node(G, np.array((xd, yd, zd)))
                        #G.add_edge(node_count, get_closest_node(G, np.array((xd, yd, zd))))

                    # else, add edge to the previous node
                    elif node_count > 1:
                        neighbor_node = node_count-1
                        G.add_edge(node_count, neighbor_node)

                    G.add_node(node_count)
                    G.nodes[node_count]['coords'] = np.array((xd, yd, zd))
                    G.nodes[node_count]['radius'] = r

                    if node_count > 1:
                        G.add_edge(node_count, neighbor_node)

                    node_count += 1

    coords = nx.get_node_attributes(G, 'coords')
    # flip y-axis for proper AP orientation
    max_y = np.max([v[1] for v in coords.values()])
    coords = {k: (coords[k][0], max_y - coords[k][1], coords[k][2]) for k in coords.keys()}

    nx.set_node_attributes(G, coords, 'coords')

    for e in G.edges():
        c1 = np.array(G.nodes[e[0]]['coords'][:2])
        c2 = np.array(G.nodes[e[1]]['coords'][:2])
        G[e[0]][e[1]]['length'] = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


    # mapping = {}
    #
    # count = 1
    #
    # for n in G.nodes():
    #     if G.degree(n) != 2:
    #         mapping[n] = count
    #         mapping[count] = n
    #         count += 1
    #     else:
    #         mapping[n] = n
    #
    # G = nx.relabel_nodes(G, mapping)

    #print(G.number_of_nodes())

    return G
    
is_right = True

for fly in range(1, 29):
    side = "R" if is_right else "L"
    
    G = trace_file_to_G(f"data/traces_L3/{fly}_Tr9{side}.traces")
    # color_plot_walk(G, f"traces/image_{fly}_{side}")
    
    # further label each cell
    # analyze branching properties
    # store them in a csv
    # find distributions for them
    # implement that into the model

    is_right = not is_right

