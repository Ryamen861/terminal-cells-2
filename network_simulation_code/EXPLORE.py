import networkx as nx
import numpy as np
import generate_networks as gn


elen = 1

# G = nx.Graph()
    
# # angle, coords, and level are made up (not in docs) so that we can access them later
# # and use them to draw the term cells
# G.add_node(0, coords=np.array((0, 0)), angle=np.pi, level = 0)
# G.add_node(1, coords=np.array((elen, 0)), angle=0, level = 0)
# G.add_edge(0, 1, level=0, length=elen)


G = gn.initialize_tri(1, elen)
# G = gn.initialize_line(1, elen)

nx.draw_spring(G, with_labels=True)
