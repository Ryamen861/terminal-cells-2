import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import time
import seaborn as sns
import ffmpeg

from generate_networks import BSARW
from helpers import get_coarsened_edge_lengths, compute_void, total_edge_length, get_num_coarsened_edges

def color_plot_walk(G, savename):

    fig, ax = plt.subplots(figsize=(2, 2))

    maxLevel = max(nx.get_edge_attributes(G, 'level').values()) + 1

    print('max level:', maxLevel)

    palette = sns.dark_palette("red", maxLevel, reverse=True)

    for e in G.edges(data=True):
        # e is a tuple that looks like this:
        # (node_connected_by_edge, other_node_connected_by_edge, dict_of_attributes)
        # the dictionary holds level, length, and ange information
        
        # find the coordinates of the two nodes connected by this edge
        c0 = G.nodes[e[0]]['coords']
        c1 = G.nodes[e[1]]['coords']

        # the level gives index for RGB value
        c = palette[e[2]['level']]
        if e[2]['level'] == 0:
            # if the level is zero, then make it blue
            # reminder: nodes of level zero are ones made in the initialization
             c = 'b'

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

max_deg = 3
elen = 1



t1 = time.time()

all_widths = []
all_heights = []

all_edges = []

all_As = []
all_Ls = []


reps = 1

size_distribution = np.random.normal(800, 200, reps)


size = 1000


# b = .5
# s = 1.001

# b = 0.1
# s = 1.0007

# b = 0.007
# s = 1.0002

# b = 0.007
# s = 1.0001

# s = 0.001
# b = 0.02

s = 0.0007
b = 0.055

# s = 0.0001
# b = 0.0001

print('s:', s, 'b:', b)


# video variables
frame_rate = 24

for rep in range(reps):

    print('size:', size)

    G = BSARW(size, elen, branch_probability = b, stretch_factor = s,
                       initial_len = 30, init = 'line', right_side_only = True)

    savename = 'Ryan_tests/N_' + str(size) + '_s_' + str(s) + '_b_' + str(b) + '_' + str(rep)
    
    #### Turn video frames into video
    video_filename = f'Ryan_video/simulation_{size}_{s}_{b}.mp4'
    
    ffmpeg\
        .input('Ryan_tests/frame_%d.png', start_number=32)\
        .output(video_filename,
                vcodec='mpeg4',
                r=str(frame_rate),
                video_bitrate='8000k')\
        .run(overwrite_output=False)
        
    ####

    B = get_num_coarsened_edges(G)
    print('number of branches:', B)

    degs = [d[1] for d in G.degree()]
    print('n1:', degs.count(1), 'n2:', degs.count(2), 'ratio:', degs.count(1)/degs.count(2))
    # print(degs.count(2))
    # print(degs.count(3))
    # print('number of branches:', B, degs.count(1) + degs.count(3) - 1)


    plt.clf()

    #plot_walk(G, sensitivity_radius, max_occupancy, latency_dist, savename, node_opts = True)

    color_plot_walk(G, savename)

    print('size:', G.number_of_nodes(), 'total length:', round(total_edge_length(G), 2))

    #voids = compute_void(G)

    #print('voids:', np.mean(voids), np.std(voids))

    #plot_walk(G, max_occupancy, latency_dist, 'corals/stretch_test_stretch_' + str(sf), node_opts = False)
    # plot_walk(G, sensitivity_radius, max_occupancy, latency_dist,
    #           #'corals/N_' + str(max_occupancy) + '_patch_' + str(rep),
    #           'corals/N_' + str(max_occupancy) + '_sf_' + str(sf) + '_L_'
    #           + str(latency_dist) + '_' + str(size) + '_tri_budding_' + str(inhibit) + '_r_' + str(sensitivity_radius) + '_' + str(rep), node_opts = False)


    #path_len = nx.shortest_path_length(G, source=0, target=30, weight='length')
    #print('new length:', round(path_len, 5), 'old length:', 30)
    #
    # H = coarsen_graph(G)
    # print('number of edges:', H.number_of_edges(), G.number_of_edges())

    N_e_coarsened = get_num_coarsened_edges(G)

    print('number of edges:', N_e_coarsened, G.number_of_edges())

    #
    # all_edges.append(H.number_of_edges())
    #
    # coords = np.array(list(nx.get_node_attributes(G, 'coords').values()))

    # G_maxs = coords.max(axis=0)
    # G_mins = coords.min(axis=0)
    # all_widths.append(G_maxs[0] - G_mins[0])
    # all_heights.append(G_maxs[1] - G_mins[1])
    #
    # voids = compute_void(G)
    # void_avg = np.mean(voids)
    # void_std = np.std(voids)
    # print('void avg:', np.mean(voids))

    # all_As.append(A)
    # all_Ls.append(L)


    #print(path_len/0.05/20)

    # nx.write_gpickle(G, 'corals/' + 'N_' + str(max_occupancy) + '_sf_' + str(sf) + '_continuous_density_L1_L_'
    #           + str(latency_dist) + '_' + str(rep) + '.gpickle')

    #nx.write_gpickle(G, 'corals/N_12_L_20_line.gpickle')

t2 = time.time()

print(round(t2-t1, 2))

print('time for', reps, 'runs of size', size, '=', round(t2-t1, 2), 'seconds')
