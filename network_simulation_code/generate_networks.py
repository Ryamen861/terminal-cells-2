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
import scipy as sci

# constants
EPSILON = sys.float_info.epsilon
GROWTH_FRACTION = 1/4

# below are 2D intersection finders
# # intersection helper
# def ccw(A, B, C):
#     return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# # return true if line segments AB and CD intersect
# def intersect(A, B, C, D):
#     return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def find_domain(initial_pt, terminal_pt, elen):
    '''Returns the domain of parametrization using magnitude restraints [0, elen].
    initial_pt is just the coordinates of the previous node.
    terminal_pt is just the coordinates of the new node.'''

    is_bound = True
    # note regarding is_bound:
    # I was thinking that " - elen " would give the upper bound, but it doesn't
    # always. So I just find both and sort them at the end
    def magnitude_function(t):
        slope_vector = np.array(terminal_pt) - np.array(initial_pt)
        
        if is_bound:
            return np.linalg.norm(slope_vector * t + initial_pt) - elen
        else:
            return np.linalg.norm(slope_vector * t + initial_pt)
        
    sol = sci.optimize.root(magnitude_function, x0=5)
    upper_bound_t = sol.x[0]
    
    is_bound = False
    sol = sci.optimize.root(magnitude_function, x0=5)
    lower_bound_t = sol.x[0]
    
    bounds = [lower_bound_t, upper_bound_t]
    bounds.sort()
    
    return tuple(bounds)

def intersect(A, B, C, D, elen):
    '''Returns a bool for whether or not two branches intersect.
    Uses vector parametrization over a domain, which is also calculated here.'''

    # p(t) = vector(AB) * t + A is the parametrization format
    # q(s) = vector(CD) * s + C
    
    does_intersect = None
    
    # find domain of t and s
    t_bounds = find_domain(A, B, elen)
    s_bounds = find_domain(C, D, elen)
    
    # find solution to system of 3 equations ( p(t) = q(s), but with x, y, z components )
    
    # two slope vectors
    ab = B - A
    cd = D - C
    
    # matrices
    coeff_matrix = np.array(
        [
            [ab[0], -cd[0]], # x values
            [ab[1], -cd[1]], # y values
            [ab[2], -cd[2]]  # z values
        ]
        )
    
    target_matrix = np.array(
        [
        [C[0] - A[0]], # x values
        [C[1] - A[1]], # y values
        [C[2] - A[2]], # z values
        ]
        )
    
    param_vars, res, rank, svs = np.linalg.lstsq(coeff_matrix, target_matrix, rcond=None)
    t, s = param_vars # however, these are best fit values, not perfect answers, so we'll have to filter through them
    
    # first, confirm that they're even within the domain
    t_within_bound = t_bounds[0] <= t and t <= t_bounds[1]
    s_within_bound = s_bounds[0] <= s and s <= s_bounds[1]
    
    if t_within_bound and s_within_bound:
        p = ab * t + A
        q = cd * s + C
        if np.linalg.norm(p - q) < EPSILON:
            # intersection
            does_intersect = True
            print("############### intersect happened")
        else:
            does_intersect = False
            
    else:
        does_intersect = False
            
    return does_intersect
    

# add new edge to node n with length elen and angle drawn from a distribution
def new_edge(G, n, elen, lev, point_tree, override = False):

    alpha1, alpha2 = get_alphas(G, n, point_tree, elen)

    # if override:
    #     buffer = np.pi / 16
    #     spread = 1
    #     alpha = uniform(-spread * buffer, spread * buffer)

    m = G.number_of_nodes()
    
    beta1 = G.nodes[n]['theta']
    new_theta = beta1 + alpha1
    
    beta2 = G.nodes[n]['phi']
    new_phi = beta2 + alpha2
    
    delta_coords = elen * np.array(
        [np.sin(new_phi) * np.cos(new_theta),
         np.sin(new_phi) * np.sin(new_theta),
         np.cos(new_phi)]
        )
    
    G.add_node(m, theta=new_theta, phi=new_phi, level = lev,
               coords = G.nodes[n]['coords'] + delta_coords)

    G.add_edge(n, m, level = lev, length = elen, theta=new_theta, phi=new_phi)

    success = True
    # check for overlaps with other edges -- if so, new addition was unsuccessful

    # find all nodes within a radius r away from m
    neibs = point_tree.query_ball_point(G.nodes[m]['coords'], 2*elen)

    # check that the new edge (n, m) does not overlap with any existing edges
    for n1 in neibs:
        for n2 in list(G.neighbors(n1)):

            A = G.nodes[n]['coords'] # coords of the previously terminal node
            B = G.nodes[m]['coords'] # coords of the new node we will add

            if n1 not in [n, m] and n2 not in [n, m]:
                C = G.nodes[n1]['coords']
                D = G.nodes[n2]['coords']

                if intersect(A, B, C, D, elen):
                    success = False

    # if the new edge has overlaps, it is removed and the
    # dock node is not chosen as a candidate again # I'm not sure if this is a true statement
    if not success:
        G.remove_node(m)

    return G, success

# add new edge to node n with length elen and angle drawn from a distribution
def new_edge_small_buffer(G, n, elen, lev, point_tree):
    '''Note: this function was not updated for 3D simulation'''

    alpha = get_alphas(G, n)

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

def check_floating_point_error(to_be_arccosed):
    if to_be_arccosed > 1:
        if math.isclose(to_be_arccosed, 1):
            return 1
        else:
            print("Then we got a real problem")
    else:
        return to_be_arccosed
    
def is_downward_growth(prev_vector, d_vector, for_theta: bool, elen):
    '''Given a previous vector (node -1 to -0) and new direction vector (0 to 1, also defined as the sum of neighboring vectors),
    gives whether the new node creates a vector with the 0 node that points up or down, with the previous vector on the x-axis.
    
    In other words, finds relative slope of two added vectors
    
    Returns:
        
        if positive slope:    True
        if negative slope:    False
        if zero slope:        None'''
        
    result = None
        
    second_index = 1 if for_theta else 2 # deciding whether to use y or z for finding theta or phi
    
    a = prev_vector[0]
    b = prev_vector[second_index]
    
    if a == 0 and b == 0:
        # if this is the case, the coeff_matrix made will be singular
        if for_theta:
            if d_vector[0] > 0:
                result = True
            else:
                result = False
        else:
            if d_vector[0] < 0:
                result = True
            else:
                result = False
        
        return result
    
    else:
        # if the coeff_matrix is not a problem
        coeff_matrix = np.array([
            [a, -b],
            [b, a]
            ])
        
        ordinate_matrix = np.array([
            [elen],
            [0]
            ])
        
        results = np.linalg.solve(coeff_matrix, ordinate_matrix)
        cos_theta = results[0][0]
        sin_theta = results[1][0]
    
        # figure out rotated coords of d_vector
        c = d_vector[0]
        d = d_vector[second_index]
        
        new_x = c * cos_theta - d * sin_theta
        new_y = c * sin_theta + d * cos_theta
            
        if new_y / new_x < 0:
            result = True
        else:
            # if it is positive or zero, just return False and we will do nothing about it
            result = False
        
        if not for_theta:
            # if the angle calcuation is for phi, phi going down is actually a positive addition to the current angle
            return not result
        else:
            return result

# pick a persistent angle for tips extension and a random angle for side budding
def get_alphas(G, n, point_tree, elen):
    '''Returns a tuple of random (theta, phi).
    The theta depends on whether or not the given node is capable of branching.'''

    buffer = 1 / 8
    # spread = 1  # 5
    # # spread = 10

    theta_1 = np.pi / 9
    
    neibs = point_tree.query_ball_point(G.nodes[n]['coords'], 2.5*elen) # 2 or 3 elen seems ok
            # however, bigger coefficient makes it more frivoly and less branchy/more longer single branches

    terminal_point = G.nodes[n]["coords"]
    final_vector = np.array([0.0, 0.0, 0.0])
    
    # add up all the vectors pointing from neighbors to current node about to bud/extend
    for node in neibs:
        initial_point = np.array(G.nodes[node]["coords"])
        new_vector = terminal_point - initial_point
        final_vector += new_vector

    # make the reference vector
    prev_node = list(G.neighbors(n))[0] # if the node whose direction we are determining right now is 1, the -1 node
    growing_node_coords = np.array(G.nodes[prev_node]["coords"])
    comp_vector = terminal_point - growing_node_coords # -1 to 0 node vector, will be used to compare to final_vector to find theta
    
    # make two 2D vectors, now can use dot product to find theta
    d_vector_2d = np.array([comp_vector[0], comp_vector[1]])
    f_vector_2d = np.array([final_vector[0], final_vector[1]])
    
    # must check for a floating point error (sometimes it comes out 1.0000000000000002 and makes arccos error)
    to_be_arccosed = np.dot(d_vector_2d, f_vector_2d) / (np.linalg.norm(d_vector_2d) * np.linalg.norm(f_vector_2d))
    to_be_arccosed = check_floating_point_error(to_be_arccosed)
    
    avoiding_theta = np.arccos(to_be_arccosed)

    # find phi, but now using the x and z components for the 2d vectors
    d_vector_2d = np.array([comp_vector[0], comp_vector[2]]) 
    f_vector_2d = np.array([final_vector[0], final_vector[2]])
    
    # check for floating point error
    to_be_arccosed = np.dot(d_vector_2d, f_vector_2d) / (np.linalg.norm(d_vector_2d) * np.linalg.norm(f_vector_2d))
    to_be_arccosed = check_floating_point_error(to_be_arccosed)
    
    avoiding_phi = np.arccos(to_be_arccosed)
    
    # however, the dot products only give positive angles, but in reality, they could be -/+
    # with the framework we decided on, so the following function will give the sign of the angles
    if is_downward_growth(comp_vector, final_vector, True, elen):
        avoiding_theta *= -1
    if is_downward_growth(comp_vector, final_vector, False, elen):
        avoiding_phi *= -1
        
    # but a lot of the time, alpha1 and alpha2 are zero because they're just chilling, no nearby nodes to avoid
    # but as they branch, there's got be some level of randomness
    # so I think it would be a good idea to combine this angle with a random angle
    # there are probably many different ways to "combine" two angles (even a distriution could be set up)
    # but just adding them could be a possible good approximation, so I will try that

    if G.degree(n) == 1:
        # pick angle for extension        
        alpha1 = uniform(-theta_1, theta_1)
    else:
        # pick angle for budding
        alpha1 = np.random.choice([-1, 1]) * np.pi / 3 # decided by looking at example data
        
    avoiding_theta *= buffer
    avoiding_phi *= buffer

    alpha2 = uniform(-np.pi / 128, np.pi / 128)
    
    return (alpha1 + avoiding_theta, alpha2 + avoiding_phi)

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

# 2 initialization functions are below (initalize_line and initialize_tri)
def initialize_line(initial_length, elen):
    '''Starts a graph with initial_length number of nodes, excluding the root
    
    '''

    # start new graph
    G = nx.Graph()
    
    # make two nodes and connect them with an edge
    G.add_node(0, coords=np.array((0, 0, 0)), theta=np.pi, phi=0, level = 0)
    G.add_node(1, coords=np.array((elen, 0, 0)), theta=0, phi=np.pi / 2, level = 0) # should any of these initial angles be random
    G.add_edge(0, 1, level=0, length=elen)

    i = 1
    
    # add new nodes until we have reached the desired length
    while i < initial_length:
        m = G.number_of_nodes()
        # the number of nodes is the number the new node will take on, since we start from 0

        # below with the alphas and deltas, we generate random delta and phi values

        alpha = uniform(-np.pi / 64, np.pi / 64) # small random deviation
        beta = G.nodes[i]['theta'] # get the angle of the latest node
        new_theta = beta + alpha # add to get new theta

        alpha2 = uniform(-np.pi / 128, np.pi / 128) # smaller range because tracheal cells are nearly flat
        beta2 = G.nodes[i]['phi']
        new_phi = alpha2 + beta2
        # there is most defnitely a problem with adding pi/2 to phi. It only goes between pi/2 and pi
        
        # create and connect the node with parameters made above
        G.add_node(m, theta=new_theta, phi=new_phi, level = 0)
        # using polar coordinates below, elen is our "roe"
        G.nodes[m]['coords'] = (G.nodes[i]['coords'][0] + elen * np.sin(new_phi) * np.cos(new_theta),
                                G.nodes[i]['coords'][1] + elen * np.sin(new_phi) * np.sin(new_theta),
                                G.nodes[i]['coords'][2] + elen * np.cos(new_phi)
                                )

        G.add_edge(i, m, level=0, length=elen, theta=new_theta, phi=new_phi)

        i += 1

    return G

def initialize_tri(initial_length, elen):

    G = nx.Graph()
    G.add_node(0, coords=np.array((0, 0, 0)), theta=np.pi, phi=0, level = 0)

    for i in [1, 2, 3]:
        new_theta = np.pi/6 + (i-1)*2*np.pi/3 # think tilted mercedes benz symbol on unit circle
        new_phi = uniform(-np.pi / 128, np.pi / 128)
        
        # add a node at tips of mercedes benz star symbol
        new_coords = elen * np.array([np.sin(new_phi) * np.cos(new_theta), np.sin(new_phi) * np.sin(new_theta), np.cos(new_phi)])
        G.add_node(i, coords=new_coords, theta=new_theta, phi=new_phi, level = 0)
        G.add_edge(0, i, level=0, length=elen)

    i = 1
    while i < initial_length:
        m = G.number_of_nodes()
        # the number of nodes is the number the new node will take on, since we start from 0

        # below with the alphas and deltas, we generate random delta and phi values

        alpha = uniform(-np.pi / 64, np.pi / 64) # small random deviation
        beta = G.nodes[i]['theta'] # get the angle of the "terminal" node
        new_theta = beta + alpha # add to get new theta

        alpha2 = uniform(-np.pi / 128, np.pi / 128)
        beta2 = G.nodes[i]['phi']
        new_phi = alpha2 + beta2
        
        # create and connect the node with parameters made above
        G.add_node(m, theta=new_theta, phi=new_phi, level = 0)
        # using polar coordinates below, elen is our "roe"
        G.nodes[m]['coords'] = (G.nodes[i]['coords'][0] + elen * np.sin(new_phi) * np.cos(new_theta),
                                G.nodes[i]['coords'][1] + elen * np.sin(new_phi) * np.sin(new_theta),
                                G.nodes[i]['coords'][2] + elen * np.cos(new_phi)
                                )

        G.add_edge(i, m, level=0, length=elen, theta=new_theta, phi=new_phi)

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
          initial_len = 75, right_side_only = False, get_intermediate_vals = False, make_video=False):

    stay_in_box = True
    stay_in_box = False

    level_num = 0

    # initialize a graph of a type depending on init
    if init == 'tri':
        G = initialize_tri(initial_len, elen)
    elif init == 'line':
        G = initialize_line(initial_len, elen)
    else:
        print('error, no initiliazation')

    print('initial condition has', G.number_of_nodes(), 'nodes')

    # in the paper, section labeled "Scaling Laws for Asymptotic Growth Regimes"
    # outlines L, Rv (abbrev as R in this code), and A as important variables 
    # that can be modeled as:        L ~ A^a and Rv ~ A^b
    intermediate_Ls = []
    intermediate_As = []
    intermediate_Rs = []
    intermediate_Bs = []

    xspan = []
    yspan = []

    step = 0
    frame_interval = 10
    frame_index = 0
    keep_adding = True

    while G.number_of_nodes() <= max_size and keep_adding:
        print(G.number_of_nodes())
        # this while loop branches/extends the graph until it has reached max size

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
        # later used to find points within a distance of a node

        if stretch_factor > 0:
            G = stretch(G, 1 + stretch_factor)

        # path_len = nx.shortest_path_length(G, source=0, target=30, weight='length')
        # print(G.number_of_nodes(), 'new length:', round(path_len, 2))



        # candidate docks must have a sufficiently small degree
        # candidate_docks = [n for n in G.nodes() if G.degree(n) < max_deg]
        # shuffle(candidate_docks)

        cands1 = [n for n in G.nodes() if G.degree(n) == 1]
        cands2 = [n for n in G.nodes() if G.degree(n) == 2]
        # if the degree is greater than 3, we should not add more to it (ruins binary tree)

        #degree_tuples = G.degree()

        shuffle(cands1)
        shuffle(cands2)

        edge_added = False

        # option to not let new branches form at the root node
        if right_side_only and init == "line":
            # this doesn't need to be considered for tri initialization because
            # the root node already has more than enough branches already (degree > 2)
            cands1.remove(0) # gets rid of node 0, the root node

        # go through list of candidate docks until one that fits all
        # docking criteria is found (density and line density requirements)
        while not edge_added:

            #print(len(cands1), len(cands2))
            
            '''
            # find the first instance of node with level = 0 (made by initialization)
            if G.number_of_nodes() / max_size <= GROWTH_FRACTION:
                # if we are a certain fraction of the way fully growed
                for node in cands2:
                    if G.nodes()[node]["level"] == 0:
                        dock = node
                        break
                
            else:
                dock = cands2.pop()                
            
            '''

            # the following block of if-statements makes it so that while we are still "young"
            # we focus more on extension than budding
            
            # BP_state is the branch probability based on the state (state as in young or old)
            if G.number_of_nodes() < 60:
                # when very young, we want some branching
                BP_state = branch_probability * 2
            elif G.number_of_nodes() < 300:
                # when medium young, we want mostly extension
                BP_state = branch_probability / 10
            else:
                # # when older, we want regular growth
                BP_state = branch_probability
                
            # this actually decides whether to bud or extend
            if random() < BP_state and len(cands2) > 0:
                dock = cands2.pop()
                
                G, edge_added = new_edge(G, dock, elen, level_num, point_tree)
                # in this case, we would be branching
            elif len(cands1) > 0:
                dock = cands1.pop()
                G, edge_added = new_edge(G, dock, elen, level_num, point_tree)
                # this case would be extending
            else:
                print("no more spaces, network stopped at", G.number_of_nodes(),
                          "nodes, but should have", max_size, "nodes")
                keep_adding = False
                break
            
            # the frame_interval makes the video more GIF-y
            if make_video and level_num % frame_interval == 0:
                color_plot_walk(G, frame_index)
                frame_index += 1

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

# the following function was copied over from test_trees.py and modified
def color_plot_walk(G, frame_index):

    fig, ax = plt.subplots(figsize=(2, 2))

    maxLevel = max(nx.get_edge_attributes(G, 'level').values()) + 1

    # print('max level:', maxLevel)

    palette = sns.dark_palette("red", maxLevel, reverse=True)

    for edge in G.edges(data=True):
        # edge is a tuple that looks like this:
        # (node_connected_by_edge, other_node_connected_by_edge, dict_of_attributes)
        # the dictionary holds level, length, and theta/phi information
        
        # find the coordinates of the two nodes connected by this edge
        c0 = G.nodes[edge[0]]['coords']
        c1 = G.nodes[edge[1]]['coords']

        # the level gives index for RGB value
        c = palette[edge[2]['level']]
        
        # the commented code above assigns color by level, or recency
        # the following code below will assign color by z coordinate
        # c = palette[int(c1[2])] # but this has negative values too, fix this
        
        if edge[2]['level'] == 0:
            # if the level is zero, then make it blue
            # reminder: nodes of level zero are ones made in the initialization
             c = 'b'

        plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color=c, linewidth = .5)

    # plt.axis('equal')

    coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    
    xs = coords[:, 0]
    ys = coords[:, 1]

    xcent = 0.5*(np.max(xs) + np.min(xs))
    ycent = 0.5*(np.max(ys) + np.min(ys))

    lim = 120

    plt.axis([xcent - lim, xcent + lim, ycent - lim, ycent + lim])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('on')
    
    plt.show()
    
    file_name = f'frames/frame_{frame_index:d}'
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.close()

