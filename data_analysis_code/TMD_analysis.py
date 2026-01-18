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
    
    return sorted(coords, key=lambda x: (-(x[1] - x[0]), x[0] + x[1]))

def color_plot_walk(G, TMD_coords, fly, side):

    savename = f"TMD_data/image_{fly}_{side}"
    fig, axs = plt.subplots(2, 2)
    
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax4.axis("off")
    ax1.text(0.05, 0.95, f'{fly}_{side}', transform=ax1.transAxes, fontsize=17, va='center_baseline', ha='center')
    
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
             
        ax1.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = 1)

    trace_coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    xs = trace_coords[:, 0]
    ys = trace_coords[:, 1]

    xcent = 0.5*(np.max(xs) + np.min(xs))
    ycent = 0.5*(np.max(ys) + np.min(ys))

    lim = 300

    ax1.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    plt.gca().set_aspect('equal', adjustable='box')
    ax1.axis('off')
    
    # for the persistence diagram
    ax2.axis(give_axes(TMD_coords)) # add a formatting function here that dynamically chooses axes
    ax2.axis("on")
    ax2.set_xlabel("Birth (distance from root)", loc="left")
    ax2.set_ylabel("Death (dist from root)")
    ax2.grid()
    
    for x, y in TMD_coords:
        ax2.scatter(x, y, color="red", s=10)
        
    # persistence barcode
    ax3.axis(give_axes(TMD_coords, barcode=True)) # add a formatting function here that dynamically chooses axes
    ax3.axis("on")
    ax3.set_xlabel("Lifetime (distance from root)")
    ax3.set_ylabel("Length of Lifetime")
    ax3.grid()
    
    # y axis is in increasing length of strand
    TMD_coords = sort_persistence_pairs(TMD_coords)

    for index in range(len(TMD_coords)):
        birth, death = TMD_coords[index]
        ax3.hlines(y=index + 1, xmin=birth, xmax=death, color='red')

    plt.savefig(savename + '.pdf', bbox_inches='tight')
    plt.close()
    
# def get_closest_node(G, n_coords):

#     all_pts = nx.get_node_attributes(G, 'coords')

#     dist_dict = {k: np.sqrt(np.sum((all_pts[k] - n_coords)**2)) for k in all_pts.keys()}

#     sorted_by_dist = dict(sorted(dist_dict.items(), key=lambda item: item[1]))
#     #closest = sorted_by_dist[0]

#     #print(list(sorted_by_dist.keys())[:5])
#     return list(sorted_by_dist.keys())[0]


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


def trace_file_to_G(filename):

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

sides = []

while fly < 29:
    
    new_side = []
    
    side = "R" if is_right else "L"
            
    G = trace_file_to_G(f"data/traces_L3/{fly}_Tr9{side}.traces")
    
    # for analysis, flip x values to positive (reflect across y axis)
    # this flipping is not done right, I think
    mod_G = copy.deepcopy(G)
    for node in mod_G.nodes():
        new_side.append(G.nodes[node]["radius"])
        

    coords = TMD(mod_G, min(list(mod_G.nodes())))
    
    # the upper left hand drawing will reflect L/R
    # the coordinates will be only the 'reflected to face the right' versions
    color_plot_walk(G, coords, fly, side)
    
    if not is_right: # if we are at left, now we can uptick, since we need fly = 1 for R and L
        fly += 1
    
    is_right = not is_right
    
    sides.append(new_side)
            
    if fly == 2: # only do the first five for now
        break













































#%% Testing

def give_axes(coords, barcode=False):
    
    if barcode:
        max_y = 0
        
        for x, y in coords:
            max_y = y if y > max_y else max_y
            
            
        return [0, max_y + 1, 0, len(coords) + 1]
        
    else:
        max_num = 0
        
        for x, y in coords:
            max_num = x if x > max_num else max_num
            max_num = y if y > max_num else max_num
            
        return [0, max_num + 1, 0, max_num + 1]
    
def sort_persistence_pairs(coords):
    """
    Sorts (birth, death) pairs by:
    1) decreasing persistence length
    2) increasing coordinate magnitude
    """
    
    return sorted(coords, key=lambda x: (-(x[1] - x[0]), x[0] + x[1]))

def color_plot_walk(G, TMD_coords, fly, side):

    savename = f"traces/TMD_{fly}_{side}"
    fig, axs = plt.subplots(2, 2)
    
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax4.axis("off")
    ax1.text(0.05, 0.95, f'{fly}_{side}', transform=ax1.transAxes, fontsize=17, va='center_baseline', ha='center')
    
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

        ax1.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = 2)

    #plt.axis('equal')
    #plt.axis('off')

    trace_coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    xs = trace_coords[:, 0]
    ys = trace_coords[:, 1]

    xcent = 0.5*(np.max(xs) + np.min(xs))
    ycent = 0.5*(np.max(ys) + np.min(ys))

    lim = 300

    ax1.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    plt.gca().set_aspect('equal', adjustable='box')
    ax1.axis('off')
    
    # for the persistence diagram
    ax2.axis(give_axes(TMD_coords)) # add a formatting function here that dynamically chooses axes
    ax2.axis("on")
    ax2.set_xlabel("Birth (distance from root)")
    ax2.set_ylabel("Death (distance from root)")
    ax2.set_xticks(np.linspace(0, 6, 7))
    ax2.set_yticks(np.linspace(0, 6, 7))
    ax2.grid()
    
    for x, y in TMD_coords:
        ax2.scatter(x, y, color="red", s=10)
        
    # persistence barcode
    ax3.axis(give_axes(TMD_coords, barcode=True)) # add a formatting function here that dynamically chooses axes
    ax3.axis("on")
    ax3.set_xlabel("Lifetime (distance from root)")
    ax3.set_ylabel("Length of Lifetime")
    ax3.set_xticks(np.linspace(0, 6, 7))
    ax3.set_yticks(np.linspace(0, 6, 7))
    ax3.grid()
    
    # y axis is in increasing length of strand
    TMD_coords = sort_persistence_pairs(TMD_coords)

    for index in range(len(TMD_coords)):
        birth, death = TMD_coords[index]
        ax3.hlines(y=index + 1, xmin=birth, xmax=death, color='red')

    plt.savefig(savename + '.pdf', bbox_inches='tight')
    plt.close()


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
color_plot_walk(k, TMD_coords, "test", "test")



