# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import gzip
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random

def f(G, n):
    '''For now, f gives the radial distance/number of edges to the root from n'''
    return nx.get_node_attributes(G, "radius")[n]

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

def trace_file_to_G(filename):

    input = gzip.open(filename, 'r')
    tree = ET.parse(input)
    root = tree.getroot()
    
    special_xs = []
    special_ys = []
    special_zs = []

    G = nx.DiGraph() # make this a directed graph
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
                    r = nDict[keys[0]]
                    
                    coords = np.array((xd, yd, zd))

                    # the following if/elif statements determine the neighbor/who to connect to
                    if first_pt and node_count > 1:
                        # the graph already has nodes, but this is the first node of the new path
                        special_xs.append(xd)
                        special_ys.append(yd)
                        special_zs.append(zd)

                        # the line below essentially creates a branch (connecting with edge happens later)
                        parent_node = get_closest_node(G, np.array((xd, yd, zd)))

                        # flipped off so it only runs once per path/branch
                        first_pt = False


                    # else: add edge to the previous node

                    elif node_count > 1:
                        # extension
                        parent_node = node_count - 1

                    G.add_node(node_count)
                    G.nodes[node_count]['coords'] = np.array((xd, yd, zd))
                    G.nodes[node_count]['radius'] = r

                    if node_count > 1:
                        # if a graph currently exists, we need to connect our 
                        # new point to the neighbor determined above
                        G.add_edge(parent_node, node_count, length=1)

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

def children_active(children, active_nodes):
    
    # edge case, there are no children
    if len(list(children)) == 0:
        return False
    
    all_children_are_active = True
    
    for child in children:
        if child not in active_nodes:
            all_children_are_active = False
            
    return all_children_are_active

def v(subtree):
    # fxs = []
    # leaves = [node for node in subtree.nodes() if subtree.degree(node) == 1]
    
    # for x in leaves:
    #     fxs.append(f(subtree, x))
    
    fxs = list(nx.get_node_attributes(subtree, "f").values())
        
    return max(fxs)

def find_possible_cms(G, children):
    print(f"length of children {len(list(children))}")
    possible_cms = []
    node_to_vcs = {}
    for child in children:
        node_to_vcs[child] = v(G.successors(child))
    
    vcs = list(node_to_vcs.values())
    print(len(vcs))
    nx.set_node_attributes(G, node_to_vcs, "v")
    max_vc = max(vcs)
    
    for child, vc in vcs:
        if vc == max_vc:
            possible_cms.append(child)
    
    return possible_cms

def TMD(G, root):
    coord_pairs = []
    
    active_nodes = np.array([node for node in G if G.degree(node) == 1 and node != root])
    
    for leaf in active_nodes:
        fls = [f(G, n) for n in active_nodes]
        fl_attributes = dict(zip(G.nodes(), fls))
        nx.set_node_attributes(G, fl_attributes, "v")
        
    while root not in active_nodes:
        print("###########")
        for leaf in active_nodes:
            
            parent = list(G.predecessors(leaf))[0]
            children = G.successors(parent)
            
            if children_active(children, active_nodes):
                possible_cms = find_possible_cms(G, children)
                Cm = random.choice(possible_cms)
                active_nodes.append(parent)
                
                for child in children:
                    active_nodes.remove(child)
                    
                    if child != Cm:
                        coord_pairs.append((v(child), f(G, parent)))
                    
                G.nodes[parent]['v'] = v(Cm)
                
    coord_pairs.append(v(root), f(G, root))
    return coord_pairs

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

nx.set_node_attributes(k, zip(list(k.nodes()), list(k.nodes())), "radius")

is_right = True

for fly in range(1, 29):
    side = "R" if is_right else "L"
    
    G = trace_file_to_G(f"data/traces_L3/{fly}_Tr9{side}.traces")
    # G = k
    # color_plot_walk(G, f"traces/image_{fly}_{side}")
    
    TMD(G, min(list(G.nodes())))

    is_right = not is_right
        
    break






