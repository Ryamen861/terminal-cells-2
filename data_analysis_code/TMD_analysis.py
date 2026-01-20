# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import gzip
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import collections
import copy

def give_axes(coords, barcode=False):
    
    if barcode:
        max_y = 0
        
        for x, y in coords:
            max_y = y if y > max_y else max_y
            
            
        return [0, max_y + 100, 0, len(coords) + 3]
        
    else:
        max_num = 0
        
        for x, y in coords:
            max_num = x if x > max_num else max_num
            max_num = y if y > max_num else max_num
            
            
        return [0, max_num + 50, 0, max_num + 50]

def sort_persistence_pairs(coords):
    """
    Sorts (birth, death) pairs by:
    1) decreasing persistence length
    2) increasing coordinate magnitude
    """
    
    for coord in coords:
        print(coord)
    
    return sorted(coords, key=lambda x: (abs(x[1] - x[0]), x[0] + x[1]))

def draw_G(G, ax):
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
             
        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = 1)

    trace_coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    xs = trace_coords[:, 0]
    ys = trace_coords[:, 1]

    xcent = 0.5*(np.max(xs) + np.min(xs))
    ycent = 0.5*(np.max(ys) + np.min(ys))

    lim = 300

    ax.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    plt.gca().set_aspect('equal', adjustable='box')
    ax.axis('off')

def show_data(G, cut_G, TMD_coords, fly, side):

    savename = f"TMD_data/image_{fly}_{side}"
    fig, axs = plt.subplots(2, 2)
    
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax2.axis("off")
    ax1.text(0.05, 0.95, f'{fly}_{side}', transform=ax1.transAxes, fontsize=17, va='center_baseline', ha='center')
    
    draw_G(G, ax1)
    draw_G(cut_G, ax2)
        
    # for the persistence diagram
    ax4.axis(give_axes(TMD_coords)) # add a formatting function here that dynamically chooses axes
    ax4.axis("on")
    ax4.set_xlabel("Birth (distance from root)", loc="left")
    ax4.set_ylabel("Death (dist from root)")
    ax4.grid()
    
    for x, y in TMD_coords:
        ax4.scatter(x, y, color="red", s=10)
        
    # persistence barcode
    ax3.axis(give_axes(TMD_coords, barcode=True))
    ax3.axis("on")
    ax3.set_xlabel("Lifetime (distance from root)")
    ax3.set_ylabel("Length of Lifetime")
    ax3.grid()

    for index in range(len(TMD_coords)):
        birth, death = TMD_coords[index]
        ax3.hlines(y=index + 1, xmin=birth, xmax=death, color='red')

    plt.savefig(savename + '.pdf', bbox_inches='tight')
    plt.close()

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


def TFG(filename):
    '''Takes in a .trace file, returns a DiGraph object constructed from trace
    TFG is short for Trace File to Graph'''

    input = gzip.open(filename, 'r')
    tree = ET.parse(input)
    root = tree.getroot()
    
    special_xs = []
    special_ys = []
    special_zs = []

    G = nx.DiGraph() # make this a directed graph
    node_count = 1
    
    # if the following for loop turns out to be wrong, copy code from previuos commit
    
    nodes = []  # list of dict(node_id, coords, radius, path_id, is_first)
    
    node_count = 1
    
    for path_id, path in enumerate(root):
        if path.tag != "path":
            continue
    
        if path.attrib["usefitted"] != "false":
            continue
    
        first = True
        for n in path.iter(tag="point"):
            nDict = n.attrib
            keys = list(nDict.keys())
    
            xd = np.round(float(nDict[keys[3]]), 4)
            yd = np.round(float(nDict[keys[4]]), 4)
            zd = np.round(float(nDict[keys[5]]), 4)
            r = float(nDict[keys[0]])
    
            nodes.append({
                "id": node_count,
                "coords": np.array((xd, yd, zd)),
                "radius": r,
                "path_id": path_id,
                "is_first": first
            })
    
            first = False
            node_count += 1
            
    G = nx.DiGraph()
    
    for n in nodes:
        G.add_node(n["id"], coords=n["coords"], radius=n["radius"])

    for i, n in enumerate(nodes):
        if i == 0:
            continue
    
        if not n["is_first"]:
            # extension
            G.add_edge(nodes[i - 1]["id"], n["id"])
        else:
            # branch
            exclude = {m["id"] for m in nodes if m["path_id"] == n["path_id"]}
            parent = get_closest_node(G, n["coords"], exclude_nodes=exclude)
            G.add_edge(parent, n["id"])

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

def trace_file_to_G(filename):

    input = gzip.open(filename, 'r')
    tree = ET.parse(input)
    root = tree.getroot()

    G = nx.DiGraph()
    node_count = 1

    nodes = []   # (id, coords, radius, path_id, is_first)

    # ---------- PASS 1: read ALL points ----------
    for path_id, path in enumerate(root):
        if path.tag != "path":
            continue
        if path.attrib["usefitted"] != "false":
            continue

        first = True
        for n in path.iter(tag="point"):
            nDict = n.attrib
            keys = list(nDict.keys())

            xd = np.round(float(nDict[keys[3]]), 4)
            yd = np.round(float(nDict[keys[4]]), 4)
            zd = np.round(float(nDict[keys[5]]), 4)
            r  = float(nDict[keys[0]])

            nodes.append({
                "id": node_count,
                "coords": np.array((xd, yd, zd)),
                "radius": r,
                "path_id": path_id,
                "is_first": first
            })

            first = False
            node_count += 1

    # ---------- ADD ALL NODES ----------
    for n in nodes:
        G.add_node(n["id"], coords=n["coords"], radius=n["radius"])

    # ---------- PASS 2: connect full paths ----------
    for i, n in enumerate(nodes):
        if i == 0:
            continue

        if not n["is_first"]:
            # extension along same path
            G.add_edge(nodes[i - 1]["id"], n["id"])
        else:
            # branch: connect first point of path to closest existing node
            exclude = {m["id"] for m in nodes if m["path_id"] == n["path_id"]}
            parent = get_closest_node(G, n["coords"], exclude_nodes=exclude)
            G.add_edge(parent, n["id"])

    # ---------- FLIP Y AXIS ----------
    coords = nx.get_node_attributes(G, 'coords')
    max_y = np.max([v[1] for v in coords.values()])
    coords = {k: (coords[k][0], max_y - coords[k][1], coords[k][2]) for k in coords}
    nx.set_node_attributes(G, coords, 'coords')

    # ---------- EDGE LENGTHS ----------
    for u, v in G.edges():
        c1 = np.array(G.nodes[u]['coords'][:2])
        c2 = np.array(G.nodes[v]['coords'][:2])
        G[u][v]['length'] = np.linalg.norm(c1 - c2)

    # ---------- PASS 3: collapse degree-2 nodes ----------
    more_to_cut = True
    while more_to_cut:
        more_to_cut = False
        # if there are no more degree 2 nodes, more_to_cut will stay False and exit loop
        for n in list(G.nodes()):
            if G.in_degree(n) == 1 and G.out_degree(n) == 1:
                parent = list(G.predecessors(n))[0]
                child  = list(G.successors(n))[0]

                # accumulate length
                new_len = G[parent][n]['length'] + G[n][child]['length']

                # reconnect
                G.add_edge(parent, child, length=new_len)

                # remove middle node
                G.remove_node(n)
                more_to_cut = True
                break

    return G

def children_active(children, active_nodes):
    
    # edge case, there are no children (not sure if this is needed at all)
    if len(list(children)) == 0:
        print("no children")
        return False
    
    all_children_are_active = True
    
    for child in children:
        if child not in active_nodes:
            all_children_are_active = False
            break
            
    return all_children_are_active

def f(G, n):
    '''For now, f gives the radial distance/number of edges to the root from n'''
    # clarification: the following line is correct, we do not need to recalculate
    # the radius to the root of G, since f is the radial distance from root R,
    # so we use the radius from the original graph
    return nx.get_node_attributes(G, "radius")[n]

def v(G, source):
    subtree = nx.dfs_tree(G, source=source)
            
    fxs = []
    for node in subtree.nodes():
        fxs.append(G.nodes[node]["f"])
            
    return max(fxs)

def find_Cm(G, children):
    possible_cms = []
    node_to_vcs = {}
        
    for child in children:
        node_to_vcs[child] = G.nodes[child]["f"]
    
    vcs = list(node_to_vcs.values())
    # nx.set_node_attributes(G, node_to_vcs, "f")
    max_vc = max(vcs)
    
    for child, vc in node_to_vcs.items():
        if vc == max_vc:
            possible_cms.append(child)
    
    return random.choice(possible_cms)

def TMD(G, root):
    coord_pairs = []
    
    active_nodes = [node for node in G if G.degree(node) == 1 and node != root]
        
    # for each leaf, give it v(l) = f(l)
    fls = [f(G, n) for n in active_nodes]
    fl_attributes = dict(zip(active_nodes, fls))
    nx.set_node_attributes(G, fl_attributes, "f")
    
    # unauthorized coding right here (me sprinkling some part that I think should be put in)
    # non_leaves = [node for node in G.nodes() if G.degree(node) != 1 and node != root]
    # all_fs = [f(G, n) for n in non_leaves]
    # fs_attr = dict(zip(G.nodes(), all_fs))
    # nx.set_node_attributes(G, fs_attr, "f")
    # f (radial distance) should be assigned for every node, whether it is a branch or leaf node
            
    while root not in active_nodes:
        for leaf in active_nodes:
            
            # print(f"We are at leaf {leaf}")
            
            parent = list(G.predecessors(leaf))[0]
            children = list(G.successors(parent))
            
            if children_active(children, active_nodes):
                Cm = find_Cm(G, children)
                
                active_nodes.append(parent)
                
                for child in children:
                    active_nodes.remove(child)
                    
                    if child != Cm:
                        coord_pairs.append((v(G, child), f(G, parent)))
                
                G.nodes[parent]['f'] = v(G, Cm)
                
    coord_pairs.append((v(G, root), f(G, root)))
    
    return coord_pairs

is_right = True

fly = 1

while fly < 29:
        
    # side = "R" if is_right else "L"
    side = "L"
            
    G, cut_G = TFG(f"data/traces_L3/{fly}_Tr9{side}.traces")     
    coords = TMD(G, min(list(cut_G.nodes())))
        
    # y axis is in increasing length of strand for persistence barcode
    TMD_coords = sort_persistence_pairs(coords)
            
    # the upper left hand drawing will reflect L/R
    # the coordinates will be only the 'reflected to face the right' versions
    show_data(G, cut_G, TMD_coords, fly, side)
    
    if not is_right: # if we are at left, now we can uptick, since we need fly = 1 for R and L
        fly += 1
    
    is_right = not is_right
                
    break












#%% Testing

k = nx.DiGraph()
for i in range(1, 11):
    k.add_node(i)
    
k.add_edge(1, 2)
k.add_edge(2, 3)
k.add_edge(3, 4)
k.add_edge(4, 5)

k.add_edge(4, 6)

k.add_edge(3, 7)

k.add_edge(2, 8)
k.add_edge(8, 9)
k.add_edge(8, 10)


radii = [0, 1, 3, 4, 6, 5, 4, 2, 3, 1]
coords = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (1, 4),
    (-1, 3),
    (-2, 2),
    (-2, 3),
    (-3, 3),
    ]

coords = np.array(coords)
coords *= 100


nx.set_node_attributes(k, dict(zip(list(k.nodes()), radii)), "radius")
nx.set_node_attributes(k, dict(zip(list(k.nodes()), coords)), "coords")


#%%
TMD_coords = TMD(k, min(list(k.nodes())))
show_data(k, k, TMD_coords, "test", "test")





