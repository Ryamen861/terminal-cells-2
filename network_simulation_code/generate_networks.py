import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import math
from random import uniform, shuffle, random, choice
import scipy.spatial as spatial
from matplotlib import patches
from helpers import total_edge_length, convex_hull_area, compute_void, number_of_branches, partial_line
import seaborn as sns

# intersection helper
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# add new edge to node n with length elen and angle drawn from a distribution
def new_edge(G, n, elen, lev, point_tree, override = False):

    alpha = get_alpha(G, n)

    # if override:
    #     buffer = np.pi / 16
    #     spread = 1
    #     alpha = uniform(-spread * buffer, spread * buffer)

    m = G.number_of_nodes()
    beta = G.nodes[n]['angle']

    angle = beta + alpha

    G.add_node(m, angle = angle, level = lev,
               coords = G.nodes[n]['coords'] + elen * np.array((np.cos(angle), np.sin(angle))))

    G.add_edge(n, m, level = lev, length = elen, angle = beta + alpha)

    success = True
    # check for overlaps with other edges -- if so, new addition was unsuccessful

    # find all nodes withing a radius r of m
    neibs = point_tree.query_ball_point(G.nodes[m]['coords'], 2*elen)

    # check that the new edge (n, m) does not overlap with any existing edges
    for n1 in neibs:
        for n2 in list(G.neighbors(n1)):

            A = G.nodes[n]['coords']
            B = G.nodes[m]['coords']

            if n1 not in [n, m] and n2 not in [n, m]:
                C = G.nodes[n1]['coords']
                D = G.nodes[n2]['coords']

                if intersect(A, B, C, D):
                    success = False

    # if the new edge has overlaps, it is removed and the
    # dock node is not chosen as a candidate again
    if not success:
        G.remove_node(m)

    return G, success

# add new edge to node n with length elen and angle drawn from a distribution
def new_edge_small_buffer(G, n, elen, lev, point_tree):

    alpha = get_alpha(G, n)

    m = G.number_of_nodes()
    beta = G.nodes[n]['angle']

    angle = beta + alpha

    G.add_node(m, angle = angle, level = lev,
               coords = G.nodes[n]['coords'] + elen * np.array((np.cos(angle), np.sin(angle))))

    G.add_edge(n, m, level = lev, length = elen, angle = beta + alpha)

    success = True
    # check for overlaps with other edges -- if so, new addition was unsuccessful

    r = 2
    global_neibs = point_tree.query_ball_point(G.nodes[m]['coords'], r*elen)

    # print(m, G.degree(n), n, global_neibs)
    #
    # if len(global_neibs) == 0:
    #     D = np.sqrt((G.nodes[n]['coords'][0] - G.nodes[m]['coords'][0])**2 +
    #                 (G.nodes[n]['coords'][1] - G.nodes[m]['coords'][1])**2)
    #
    #     print(G.nodes[n]['coords'], G.nodes[m]['coords'], D)
    #
    #     color_plot_walk(G, 'corals/problem_' + str(m))


    #graph_neibs = G.neighbors(m)

    local_neibs = nx.single_source_shortest_path(G, m, cutoff = 2*r*elen)

    #paths_to_neibs = [nx.dijkstra_path(G, m, t, weight='length') for t in global_neibs]

    #print(m, global_neibs, list(local_neibs.keys()))

    for p in local_neibs:
        if p in global_neibs:
            global_neibs.remove(p)

    # if the new edge has overlaps, it is removed
    if len(global_neibs) > 0:
        #print('nonzero:', global_neibs)

        #color_plot_walk(G, 'corals/failed_' + str(m), special = global_neibs[0])
        G.remove_node(m)

    return G, success

# add new edge to node n with length elen and angle drawn from a distribution
def new_long_edge(G, n, elen, lev, point_tree):

    global_success = True

    G, success = new_edge(G, n, elen, lev, point_tree)

    # G, success = new_edge(G, G.number_of_nodes()-1, elen, lev, point_tree, override = True)
    # G, success = new_edge(G, G.number_of_nodes()-1, elen, lev, point_tree, override = True)

    return G

# pick a persistent angle for tips extension and a random angle for side budding
def get_alpha(G, n):

    buffer = np.pi / 16
    spread = 1  # 5
    # spread = 10

    theta_1 = np.pi / 9

    if G.degree(n) == 1:
        #alpha = uniform(-spread * buffer, spread * buffer)
        alpha = uniform(-theta_1, theta_1)
    else:
        # pick up or down sprouting direction by the sign
        #alpha = np.random.choice([-1, 1]) * uniform(np.pi / 2 - spread * buffer, np.pi / 2 + spread * buffer)

        alpha = np.random.choice([-1, 1]) * np.pi /2#5/6



    return alpha

# get number of nodes within radius r of node n
def get_node_occupancy(G, n, r, point_tree):
    return len(point_tree.query_ball_point(G.nodes[n]['coords'], r))

# get number of nodes within radius r of node n
def get_node_occupancy_continuous(G, n, r, point_tree):
    
    captured_pts = point_tree.query_ball_point(G.nodes[n]['coords'], r, p=2)

    sum_len = 0
    counted_edges = []

    for i in captured_pts:
        for j in G.neighbors(i):

            if (j, i) not in counted_edges:

                if j in captured_pts:
                    sum_len += G[i][j]['length']
                else:
                    sum_len += partial_line(G.nodes[n]['coords'], r,
                                            G.nodes[i]['coords'], G.nodes[j]['coords'])

                counted_edges.append((i, j))

    return sum_len

# from node n, find the closest node m with degree 1 or 3
# distance is measured by edge length
def get_latency_dist(G, n, L):

    length, path = nx.multi_source_dijkstra(G, [n], target = None, cutoff=L, weight="length")

    for m in length.keys():
        if G.degree(m) != 2:
            return length[m]

    return L

def stretch(G, alpha):

    coords = nx.get_node_attributes(G, 'coords')
    coords.update({k: alpha * np.array(coords[k]) for k in coords.keys()})
    nx.set_node_attributes(G, coords, 'coords')

    e_lens = nx.get_edge_attributes(G, 'length')
    e_lens.update({k: alpha * e_lens[k] for k in e_lens.keys()})
    nx.set_edge_attributes(G, e_lens, 'length')

    # for e in G.edges():
    #     G[e[0]][e[1]]['length'] = alpha*G[e[0]][e[1]]['length']

    return G

# 2 initalization functions are below (initalize_line and initialize_tri)
def initialize_line(initial_length, elen):

    G = nx.Graph()
    
    G.add_node(0, coords=np.array((0, 0)), angle=np.pi, level = 0)
    G.add_node(1, coords=np.array((elen, 0)), angle=0, level = 0)
    G.add_edge(0, 1, level=0, length=elen)

    i = 1
    while i < initial_length:
        alpha = uniform(-np.pi / 64, np.pi / 64)

        m = G.number_of_nodes()
        beta = G.nodes[i]['angle']

        angle = beta + alpha

        G.add_node(m, angle=angle, level = 0)
        G.nodes[m]['coords'] = (G.nodes[i]['coords'][0] + elen * np.cos(angle),
                                G.nodes[i]['coords'][1] + elen * np.sin(angle))

        G.add_edge(i, m, level=0, length=elen, angle=beta + alpha)

        i += 1

    return G

def initialize_tri(initial_length, elen):

    G = nx.Graph()
    G.add_node(0, coords=np.array((0, 0)), angle=np.pi, level = 0)

    for i in [1, 2, 3]:
        ang = np.pi/6 + (i-1)*2*np.pi/3
        G.add_node(i, coords=elen*np.array((np.cos(ang), np.sin(ang))), angle=ang, level = 0)
        G.add_edge(0, i, level=0, length=elen)

    i = 1
    while i < initial_length:
        alpha = uniform(-np.pi / 32, np.pi / 32)

        m = G.number_of_nodes()
        beta = G.nodes[i]['angle']

        angle = beta + alpha

        G.add_node(m, angle=angle, level = 0)
        G.nodes[m]['coords'] = (G.nodes[i]['coords'][0] + elen * np.cos(angle),
                                G.nodes[i]['coords'][1] + elen * np.sin(angle))

        G.add_edge(i, m, level=0, length=elen, angle=beta + alpha)

        i += 1

    return G

# max_size: stop the growth when this number of edges is reached, 
#       or earlier if no more moves are possible
# elen: size of every new edge
# sensitivity radius: sets radius of density sensing
# max_occupancy: sets density limit
# latency_dist: allowed distance from degree 1 or 3 vertex that a new bud can form
# stretch_factor: stretching growth factor

box_lims = [0, 60, -10, 20]

def BSARW(max_size, elen, branch_probability = .1, stretch_factor = 0, init = 'tri', max_deg = 3, 
          initial_len = 75, right_side_only = False, get_intermediate_vals = False):

    stay_in_box = True
    stay_in_box = False

    level_num = 0

    if init == 'tri':
        G = initialize_tri(initial_len, elen)
    elif init == 'line':
        G = initialize_line(initial_len, elen)
    else:
        print('error, no initiliazation')

    print('initial condition has', G.number_of_nodes(), 'nodes')


    intermediate_Ls = []
    intermediate_As = []
    intermediate_Rs = []
    intermediate_Bs = []

    xspan = []
    yspan = []

    step = 0
    keep_adding = True

    while G.number_of_nodes() <= max_size and keep_adding:

        #f = 0.75
        # box_lims = [0, W, -20 - (1-f)*0.1*step, 20 + f*0.1*num]
        W = 50 + 0.075*level_num
        H = 20 + 0.1*level_num
        box_lims = [0, W, -H/4, 3*H/4]

        level_num += 1

        if get_intermediate_vals:
            intermediate_Ls.append(total_edge_length(G))
            intermediate_As.append(convex_hull_area(G))
            intermediate_Rs.append(np.mean(compute_void(G)))
            intermediate_Bs.append(number_of_branches(G))

            # spans = get_spans(G)
            #
            # xspans.append(spans[0])
            # yspans.append(spans[1])


        point_tree = spatial.cKDTree(list(nx.get_node_attributes(G, 'coords').values()))

        if stretch_factor > 0:
            G = stretch(G, 1 + stretch_factor)

        # path_len = nx.shortest_path_length(G, source=0, target=30, weight='length')
        # print(G.number_of_nodes(), 'new length:', round(path_len, 2))



        # candidate docks must have a sufficiently small degree
        # candidate_docks = [n for n in G.nodes() if G.degree(n) < max_deg]
        # shuffle(candidate_docks)

        cands1 = [n for n in G.nodes() if G.degree(n) == 1]
        cands2 = [n for n in G.nodes() if G.degree(n) == 2]

        #degree_tuples = G.degree()

        shuffle(cands1)
        shuffle(cands2)

        edge_added = False

        # option to not let new branches form at the base
        if right_side_only:
            cands1.remove(0)

        # go through list of candidate docks until one that fits all
        # docking criteria is found (density and line density requirements)
        while not edge_added:

            #print(len(cands1), len(cands2))

            if random() < branch_probability and len(cands2) > 0:
                dock = cands2.pop()
                G, edge_added = new_edge(G, dock, elen, level_num, point_tree)
            elif len(cands1) > 0:
                dock = cands1.pop()
                G, edge_added = new_edge(G, dock, elen, level_num, point_tree)
            else:
                print("no more spaces, network stopped at", G.number_of_nodes(),
                          "nodes, but should have", max_size, "nodes")
                keep_adding = False
                break

        # frame = 50
        # if level_num % frame == 1:
        #     color_plot_walk(G, 'gifs/s_' + str(stretch_factor) + \
        #                     '_b_' +  str(branch_probability) + '_x' + str(frame) + '/' + str(level_num))


    return G

def BSARW_no_tip_ext(max_size, elen, branch_factor = 1, stretch_factor = 1, 
                     init = 'tri', max_deg = 3, initial_len = 75, right_side_only = False):

    if init == 'tri':
        G = initialize_tri(initial_len, elen)
    elif init == 'line':
        G = initialize_line(initial_len, elen)
    else:
        print('error, no initiliazation')

    print('initial condition has', G.number_of_nodes(), 'nodes')

    level_num = 1

    while G.number_of_nodes() <= max_size:

        point_tree = spatial.cKDTree(list(nx.get_node_attributes(G, 'coords').values()))

        if stretch_factor > 1:
            G = stretch(G, stretch_factor)


        #candidate_docks = [n for n in G.nodes() if G.degree(n) == 2]
        candidate_docks = [n for n in G.nodes() if G.degree(n) < max_deg]

        candidate_docks.remove(0)

        # if len(candidate_docks) == 0:
        #     plot_walk(G, sensitivity_radius, max_occupancy, latency_dist, 'corals/finished_net_' + str(G.number_of_nodes()) + '.png')


        if random() > branch_factor:
            dock = choice(candidate_docks)
            #G = new_long_edge(G, dock, elen, level_num, point_tree)
            G, success = new_edge(G, dock, elen, level_num, point_tree)
            if success:
                level_num += 1

    return G

# G: coral tree network
def plot_walk(G, sensitivity_radius, max_occupancy, latency_dist, savename, dlim = 1, node_opts = False,
              box_lims = []):

    fig, ax = plt.subplots(figsize=(3, 3))

    #print(nx.get_node_attributes(G, 'occupancy'))

    for e in G.edges(data = True):

        c0 = G.nodes[e[0]]['coords']
        c1 = G.nodes[e[1]]['coords']

        c = 'k'

        # local_rho = G.nodes[e[1]]['occupancy']#/G.nodes[e[1]]['neighborhood_size']

        #print(e[1], local_rho, G.nodes[e[1]]['neighborhood_size'])

        #print(len(list(G.neighbors(e[0]))), len(list(G.neighbors(e[1]))))

        # if G.nodes[e[0]]['occupancy'] > dlim and G.nodes[e[1]]['occupancy'] > dlim and \
        #         (1 <= G.nodes[e[1]]['latency_dist'] and G.nodes[e[1]]['latency_dist'] <= 5) and \
        #         (1 <= G.nodes[e[0]]['latency_dist'] and G.nodes[e[0]]['latency_dist'] <= 5) and \
        #         len(list(G.neighbors(e[0]))) < 3 and len(list(G.neighbors(e[1]))) < 3:
        #
        #     c = 'r'

        if e[2]['level'] == 0: c = 'b'
        plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color = c, linewidth = 1.5, zorder = 0)

    if node_opts:

        point_tree = spatial.cKDTree(list(nx.get_node_attributes(G, 'coords').values()))

        for n in G.nodes():

            occupancy = get_node_occupancy_continuous(G, n, sensitivity_radius, point_tree)

            if occupancy > max_occupancy:
                plt.scatter(G.nodes[n]['coords'][0], G.nodes[n]['coords'][1], c='r', s=2, zorder=1)

                #print(occupancy)

            # ident = round(occupancy, 1)
            # # ident = n
            # buff = 0#.007
            # plt.text(G.nodes[n]['coords'][0] + buff, G.nodes[n]['coords'][1] + buff,
            #          ident, c='r', size=1, zorder = 10)
            #
            # latency = get_latency_dist(G, n, latency_dist)
            #
            # if G.degree(n) < 3 and occupancy < max_occupancy and (latency >= latency_dist or latency == 0):
            #     plt.scatter(G.nodes[n]['coords'][0], G.nodes[n]['coords'][1], c='cornflowerblue', s=2, alpha = 0.5, zorder=1)

            # if G.degree(n) == 1:
            #     c = G.nodes[n]['coords']
            # 
            #     patch = patches.Circle((c[0], c[1]), radius = sensitivity_radius, facecolor = 'none',
            #                            edgecolor = (0.62, 0, 0), linewidth = .2)
            #     ax.add_patch(patch)

            # elif get_latency_dist(G, n, latency_dist) >= latency_dist:
            #     plt.scatter(G.nodes[n]['coords'][0], G.nodes[n]['coords'][1], c='r', s=2, zorder=5)

            # elif G.nodes[n]['active']:
            #     plt.scatter(G.nodes[n]['coords'][0], G.nodes[n]['coords'][1], c='r', s=2, zorder=5)

    if len(box_lims) > 0:
        fence = patches.Rectangle((box_lims[0], box_lims[2]),
                                  box_lims[1] - box_lims[0],
                                  box_lims[3] - box_lims[2],
                                   linewidth=1, edgecolor='b',
                                   facecolor='b', alpha=0.1)
        ax.add_patch(fence)
    
    
    
    

    # for n in G.nodes():
    #     #ident = round(get_latency_dist(G, n, latency_dist), 1)
    #     ident = round(get_node_occupancy(G, n, sensitivity_radius, point_tree), 2)
    #     #ident = n
    #     buff = 0.007
    #     plt.text(G.nodes[n]['coords'][0] + buff, G.nodes[n]['coords'][1] + buff,
    #              ident, c='r', size = 1)


    # for n in G.nodes(data=True):
    #     c = G.nodes[n[0]]['coords']
    #     d = n[1]['latency_dist']
    #     d = n[1]['occupancy']
    #     # d_old = n[1]['latency_dist_old']
    #     # if d !=  d_old:
    #     #     print('problem node:', n[0], d, d_old)
    # 
    #     label = d
    #     #label = n[0]
    # 
    #     #plt.text(c[0], c[1], label, color='r', fontsize=2)
    # 
    #     if G.degree(n[0]) == 1:
    #         patch = patches.Circle((c[0], c[1]), radius = sensitivity_radius, facecolor = 'none',
    #                                edgecolor = (0.62, 0, 0), linewidth = .2)
    #         ax.add_patch(patch)
    # 
    #         if d >= max_occupancy: col = (0.62, 0, 0)
    #         else: col = 'b'
    # 
    #         plt.scatter(c[0], c[1], color = col, s = 2)
    #             #plt.text(c[0], c[1], label, color='r', fontsize=2)
    
    

    #     s = n[1]['neighborhood_size']
    #     occu = n[1]['occupancy']
    #     d = n[1]['local_density']
    #
    #     print ((1+occu)/s, d)
    #
    #     # plt.text(c[0], c[1], n, color = 'r', fontsize=6)
    #     #plt.text(c[0], c[1], G.nodes[n]['latency_dist'], color = 'r', fontsize=6)
    #     plt.text(c[0], c[1], np.round(G.nodes[n]['local_density'], 2), color = 'r', fontsize=6)

    # c = 0.5
    # plt.xlim(c - 0.5, c + 0.5)
    # plt.ylim(c - 0.5, c + 0.5)

    #plt.title('Local Neighbor Limit = ' + str(dlim))

    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])

    # plt.xlim(-10, 20)
    # plt.ylim(-15, 15)

    print('plotting', savename)

    # box_lim = 10
    # plt.xlim(-box_lim, box_lim)
    # plt.ylim(-box_lim, box_lim)

    #plt.axis('off')
    #plt.axis('equal')
    # plt.xlim(-20, 220)
    # plt.ylim(-80, 200)

    # plt.xlim(-40, 200)
    # plt.ylim(-120, 120)
    #
    # plt.xlim(-100, 340)
    # plt.ylim(-240, 240)

    plt.axis('equal')

    plt.savefig(savename + '.png', bbox_inches = 'tight', dpi = 300)
    plt.close()

    #nx.write_gpickle(G, savename + "_saved.gpickle")



def color_plot_walk(G, savename, special = 0):

    fig, ax = plt.subplots(figsize=(4, 4))

    palette = sns.dark_palette("red", G.number_of_nodes(), reverse=True)

    for e in G.edges(data=True):

        #print(e, e[2]['level'])

        c0 = G.nodes[e[0]]['coords']
        c1 = G.nodes[e[1]]['coords']


        c = palette[e[2]['level']]
        #c = 'k'

        if e[2]['level'] == 0:
             c = 'b'
            
        c = 'k'

        plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = .5)

    # m = G.number_of_nodes() - 1
    # c0 = G.nodes[m]['coords']
    # plt.scatter(c0[0], c0[1], color = 'k')
    # 
    # if special > 0:
    #     c0 = G.nodes[special]['coords']
    #     plt.scatter(c0[0], c0[1], color='b')
    # 
    # print(list(G.neighbors(m)))

    plt.xlim(-180, 380)
    plt.ylim(-280, 280)

    #plt.axis('equal')

    plt.savefig(savename + '.pdf', bbox_inches='tight', dpi=300)
    plt.close()