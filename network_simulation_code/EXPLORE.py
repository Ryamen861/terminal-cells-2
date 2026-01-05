import networkx as nx
import numpy as np
import generate_networks as gn
from random import uniform, shuffle, random, choice
import scipy as sci
import math

PHI_BOUND_VAR = 2

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

elen = 1
G = initialize_line(30, elen)
x = G.neighbors(3)

#%%

def find_sign_angle(prev_vector, d_vector, elen, for_theta: bool):
    '''Given a previous vector (node -1 to -0) and new direction vector (0 to 1, also defined as the sum of neighboring vectors),
    gives whether the new node creates a vector with the 0 node that points up or down, with the previous vector on the x-axis.
    
    In other words, finds relative slope of two added vectors
    
    Returns:
        
        if positive slope:    True
        if negative slope:    False
        if zero slope:        None'''
        
    second_index = 1 if for_theta else 2

    a = prev_vector[0]
    b = prev_vector[second_index]
    
    coeff_matrix = np.array([
        [a, -b],
        [b, a]
        ])
    
    ordinate_matrix = np.array([
        [elen],
        [0]
        ])
    
    print(coeff_matrix)
    print("\n\n\n")
    
    results = np.linalg.solve(coeff_matrix, ordinate_matrix)
    cos_angle = results[0][0]
    sin_angle = results[1][0]

    # figure out rotated coords of d_vector
    c = d_vector[0]
    d = d_vector[second_index]
    
    new_x = c * cos_angle - d * sin_angle
    new_y = c * sin_angle + d * cos_angle
    
    if new_y / new_x < 0:
        return False
    else:
        # if it is positive or zero, just return True and we will do nothing about it
        return True
    
# def find_sign_phi(prev_vector, d_vector, theta, elen):
#     prev_vector *= -1 # switch its direction for cross product
    
#     # depending on sign of theta, cross product order must be different
#     if theta > 0:
#         upward_vector = np.cross(d_vector, prev_vector)
        
#     elif theta < 0:
#         upward_vector = np.cross(prev_vector, d_vector)
#     else:
#         # if theta = 0
#         pass
    
# def find_sign_phi_2(prev_vector, d_vector, theta, elen):
#     slope_vector = prev_vector + d_vector
#     x = slope_vector[0]
#     y = slope_vector[1]
    
#     phi1 = np.arcsin(x / (elen * np.cos(theta)))
#     phi2 = np.arcsin(y / (elen * np.sin(theta)))
    
#     print(phi1, phi2)


a = np.array([0, 1, 0])
b = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 1])
b = b / np.linalg.norm(b)

find_sign_angle(a, b, 1, for_theta=False)


#%%

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
    
    # the following cases brings about a singular matrix so they will be handled here        
    if a == 0 and b == 0:
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

a = np.array([0, 1, 0])
b = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 1])
b = b / np.linalg.norm(b)

downward_phi = is_downward_growth(a, b, False, 1)
downward_theta = is_downward_growth(a, b, True, 1)

print(downward_theta, downward_phi)

    #%%
    
def check_floating_point_error(to_be_arccosed):
    if to_be_arccosed > 1:
        if math.isclose(to_be_arccosed, 1):
            return 1
        else:
            print("Then we got a real problem")
    else:
        return to_be_arccosed

# pick a persistent angle for tips extension and a random angle for side budding
def get_alphas(G, n, point_tree, elen):
    '''Returns a tuple of random (theta, phi).
    The theta depends on whether or not the given node is capable of branching.'''

    # buffer = np.pi / 16
    # spread = 1  # 5
    # # spread = 10

    theta_1 = np.pi / 9
    
    neibs = point_tree.query_ball_point(G.nodes[n]['coords'], 2*elen)
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
        # pick angle for side budding        
        alpha1 = uniform(-theta_1, theta_1)
    else:
        # pick angle for tip extension
        alpha1 = np.random.choice([-1, 1]) * np.pi / 2 # why only these values though

    alpha2 = uniform(-np.pi / 128, np.pi / 128)
    
    return (alpha1 + avoiding_theta, alpha2 + avoiding_phi)


