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
import pandas as pd

from convert_trace_to_network import trace_file_to_G

num_of_aries = {
                1:1, # by design this should be one since the longest
                2:0,
                3:0,
                4:0,
                5:0,
                }

total_properties_dataset = []
ary_properties = {
                    1: {"length": 0, "offshoots": 0},
                    2: {"length": 0, "offshoots": 0},
                    3: {"length": 0, "offshoots": 0},
                    4: {"length": 0, "offshoots": 0}
    }


total_fork_dist_dataset = []
avg_dist_bw_forks = {
                1:0,
                2:0,
                3:0,
                4:0,
                5:0,
    }

def color_plot_walk(G, savename):

    fig, ax = plt.subplots(figsize=(2, 2))

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

def collapse(dataset: list):
    
    formatted_data = {
        
        }
    
    ary_keys = []
    
    for data_dict in dataset:
        ary_keys.append(list(data_dict.keys())[0])
        
    max_branch_order = max(ary_keys)
    
    keys = ["num_of_nodes", "fork distance", "number of offshoots"]
    
    for ary in range(1, max_branch_order + 1):

        branch_attributes = []
        
        for key in keys:

            attribute_collection = []

            for data_dict in dataset:
                if list(data_dict.keys())[0] == ary:
                    # if we find the order type of a branch that we're looking for
                    # collect its info into attribute_collection
                    
                    # print(f"before concatenation for {key}: {data_dict[ary][key]}")
                    
                    if key == "fork distance":
                        # list concatenation
                        
                        attribute_collection.extend(data_dict[ary][key])
                    else:
                        # add number to list
                        attribute_collection.append(data_dict[ary][key])

            to_be_averaged = np.array(attribute_collection)
            
            branch_attributes.append(np.mean(to_be_averaged))
        
        formatted_data[ary] = branch_attributes # order of objects is length, fork distance, num of offshoots
        
    return formatted_data, max_branch_order
        

def find_properties(G, ary=1, dataset: list = []):
    '''Finds the following properties of a Graph and places them in the lists at the top of the page
    The number 
    The lengths of the branches
    The number of offshoots of the branches
    The distance between the offshoots of the branches
    
    of each branch type (the orders: primary, secondary, tertiary)
    
    '''
    
    print(ary)
    
    # catch edge case when G is just one node
    if len(G.nodes()) == 1:
        dataset.append({
            ary: {
                "num_of_nodes" : 1,
                "fork distance" : [],
                "number of offshoots": 0,
                
                }
            })
    else:

        # finding the longest path
        tip_index = None
        all_nodes = list(G.nodes())
        all_nodes.sort()


        for node in all_nodes[1:]: # we want to avoid the first one (root node)
            if G.degree(node) == 1:
                tip_index = all_nodes.index(node)
                break
                    
        max_path = all_nodes[:tip_index + 1]
        # print(all_nodes, max_path)
        max_length = len(max_path)
        # print(f"max path {max_path}")
        
        # find brancher nodes
        branchers = np.array([node for node in max_path if G.degree(node) == 3])
        
        # print(f"branchers {branchers}")
        
        distances_bw_branchers = []
        nodes = list(G.nodes())
    
        for i in range(len(branchers) - 1):
            node_of_branched = branchers[i]
            next_node_of_branched = branchers[i + 1]
            
            distance = nodes.index(next_node_of_branched) - nodes.index(node_of_branched)
            distances_bw_branchers.append(distance)
        
        filtered_path = [node for node in max_path if node not in branchers]
        G.remove_nodes_from(filtered_path)
        # delete the branch we just looked at
        
        # case where two branchers are right next to each other, separate them
        for index in range(0, len(branchers) - 1):
            if branchers[index + 1] - branchers[index] == 1:
                # if right next to each other
                G.remove_edge(branchers[index + 1], branchers[index])
                                
        # upperdary is a mix between upper and second-ary, tertiary notation
        # since we may have multiple secondary brances on one primary branch, because we removed the primary branch
        # we have disconnected secondary branches
        upperdary_graphs = [G.subgraph(sub).copy() for sub in nx.connected_components(G)]
        
        if ary > 1:
            num_of_nodes = max_length - 1 # to exclude the root that is part
            # of the branch below
        else:
            num_of_nodes = max_length
            
        # below is not needed because that means it has no branches. If it has no branches
        # we don't want it to affect the average. The average should describe the average
        # distance between branches SHOULD THEY EXIST
        # if len(distances_bw_branchers) == 0:
        #     distances_bw_branchers.append(0)
        
        # record the data
        dataset.append({
            ary: {
                "num_of_nodes" : num_of_nodes,
                "fork distance" : distances_bw_branchers,
                "number of offshoots": len(branchers),
                
                }
            })
        
        # find the properties for the branches that are offshooting from the one we just observed
        # print(f"number of now disconnected graphs {len(upperdary_graphs)}")
        for upperdary in upperdary_graphs:
            find_properties(upperdary, ary=ary + 1, dataset=dataset)
            
    return dataset
        
    
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

                first_pt = True

                for n in path.iter(tag='point'):

                    nDict = n.attrib
                    keys = list(nDict.keys())

                    # print(nDict)

                    xd = np.round(float(nDict[keys[3]]), 4)
                    yd = np.round(float(nDict[keys[4]]), 4)
                    zd = np.round(float(nDict[keys[5]]), 4)
                    r = float(nDict[keys[0]])
                    
                    coords = np.array((xd, yd, zd))

                    # if first node and node not 1, add edge to the closest node in existing network

                    if first_pt and node_count > 1:
                        special_xs.append(xd)
                        special_ys.append(yd)
                        special_zs.append(zd)
                        first_pt = False

                        neighbor_node = get_closest_node(G, np.array((xd, yd, zd)))
                        G.add_edge(node_count, get_closest_node(G, np.array((xd, yd, zd))), length=1)

                    # else: add edge to the previous node

                    elif node_count > 1:
                        neighbor_node = node_count-1
                        G.add_edge(node_count, node_count-1, length=1)

                    G.add_node(node_count)
                    G.nodes[node_count]['coords'] = np.array((xd, yd, zd))
                    G.nodes[node_count]['radius'] = r

                    if node_count > 1:
                        G.add_edge(node_count, neighbor_node, length=1)

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

    return G
#%%
k = nx.DiGraph()
for i in range(1, 18):
    k.add_node(i)
    
k.add_edge(1, 2)
k.add_edge(2, 3)
k.add_edge(3, 4)
k.add_edge(4, 5)
k.add_edge(5, 6)
k.add_edge(6, 7)

k.add_edge(4, 8)
k.add_edge(8, 9)
k.add_edge(8, 10)

k.add_edge(2, 11)
k.add_edge(11, 12)
k.add_edge(12, 13)
k.add_edge(12, 14)
k.add_edge(11, 15)

k.add_edge(14, 16)
k.add_edge(14, 17)

# nx.draw_forceatlas2(k, with_labels=True)

# raw_data = find_properties(k)
# data = collapse(raw_data)

    
#%%
is_right = True

for fly in range(1, 29):
    side = "R" if is_right else "L"
    
    G = trace_file_to_G(f"data/traces_L3/{fly}_Tr9{side}.traces")
    color_plot_walk(G, f"traces/image_{fly}_{side}")
            
    # further label each cell        
    # analyze branching properties
    
    # data, max_branch_order = collapse(find_properties(G))
    
    # for order in range(1, max_branch_order + 1):
        
    #     length_data = data[order][0]

    #     df = pd.read_csv(f'{order}_lengths.csv')
    #     new_data = pd.DataFrame([{"node count": length_data}])
    #     df = pd.concat([df, ], ignore_index = True)
    
    # df = pd.DataFrame(data)
    # df.to_csv("data.csv")
    
    # store them in a csv
    # find distributions for them
    # implement that into the model

    is_right = not is_right
    
    find_properties(G)


#%% Topology stuff

import networkx as nx

def remove_degree_2_nodes(G):
    """
    Remove nodes with in-degree = 1 and out-degree = 1,
    preserving directed connectivity between endpoints.
    """
    if not G.is_directed():
        raise ValueError("Graph must be directed")

    H = nx.DiGraph()
    
    # Nodes to keep
    keep = [
        n for n in G.nodes
        if not (G.in_degree(n) == 1 and G.out_degree(n) == 1)
    ]
    H.add_nodes_from(keep)

    for u in keep:
        for v in G.successors(u):
            curr = v
            prev = u

            # Walk through degree-2 chain
            while (
                curr not in keep and G.in_degree(curr) == 1 and G.out_degree(curr) == 1):
                nxt = next(G.successors(curr))
                prev, curr = curr, nxt

            H.add_edge(u, curr)

    return H


def renumber_nodes_directed(G, root):
    """
    Renumber nodes in a directed acyclic graph so that:
    - Closer to root → smaller number
    - Nodes on longer downstream paths → larger number
    - Ties broken by original node id
    """

    if not G.is_directed():
        raise ValueError("Graph must be directed")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG")

    # 1. Directed distance from root
    depth = nx.single_source_shortest_path_length(G, root)

    # 2. Longest downstream path (DP on DAG)
    topo = list(nx.topological_sort(G))
    subtree_height = {n: 0 for n in G.nodes}

    for u in reversed(topo):
        heights = [
            1 + subtree_height[v]
            for v in G.successors(u)
        ]
        subtree_height[u] = max(heights, default=0)

    # 3. Sort nodes
    ordered_nodes = sorted(
        G.nodes,
        key=lambda n: (
            depth.get(n, float("inf")),   # closer to root first
            subtree_height[n],             # shorter paths first
            n                              # original label
        )
    )

    # 4. Relabel
    mapping = {node: i + 1 for i, node in enumerate(ordered_nodes)}
    G_new = nx.relabel_nodes(G, mapping)

    return G_new # , mapping

def draw_binary_tree(G, title):
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, pos, with_labels=True, node_color='orange', node_size=500, font_size=10)
    plt.axis('off') # Hide the axis ticks and labels
    plt.title(f"{title}")
    plt.show()

print(k.nodes())

simplified = remove_degree_2_nodes(k)
draw_binary_tree(simplified, "simplified")
print(simplified.nodes())

renumbered = renumber_nodes_directed(simplified, min(list(simplified.nodes())))
draw_binary_tree(renumbered, "renumbered")
print(renumbered.nodes())

#%%

a = nx.DiGraph()
a.add_nodes_from(list(range(1, 9)))
a.add_edges_from([
    (1, 2),
    (2, 4),
    (2, 3),
    (3, 5),
    (3, 6),
    (6, 8),
    (6, 7),
    ])

b = nx.DiGraph()
b.add_nodes_from(list(range(1, 9)))
b.add_edges_from(
    [
    (1, 2),
    (2, 4),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 8),
    (5, 7)
    ]
    )

graphs = [a, b]
matrices = []
for index in range(len(graphs)):
    graph = graphs[index]
    simplified = remove_degree_2_nodes(graph)

    renumbered = renumber_nodes_directed(simplified, min(list(simplified.nodes())))
    draw_binary_tree(renumbered, f"{index + 1}")
    
    new_AM = nx.to_numpy_array(renumbered)
    # print(new_AM)
    matrices.append(new_AM)

print(matrices[0] == matrices[1])





