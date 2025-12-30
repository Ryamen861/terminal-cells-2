import networkx as nx
import numpy as np
import generate_networks as gn
from random import uniform, shuffle, random, choice
import scipy as sci



elen = 1

# G = nx.Graph()
    
# # angle, coords, and level are made up (not in docs) so that we can access them later
# # and use them to draw the term cells
# G.add_node(0, coords=np.array((0, 0)), angle=np.pi, level = 0)
# G.add_node(1, coords=np.array((elen, 0)), angle=0, level = 0)
# G.add_edge(0, 1, level=0, length=elen)

A = np.array([0, 0, 0])
B = np.array([1, 0, 0])

C = np.array([1, 1, 1])
D = np.array([1, 2, 1])

gn.intersect(A, B, C, D, 2)