import networkx as nx
import numpy as np
import generate_networks as gn
from random import uniform, shuffle, random, choice
import scipy as sci

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



