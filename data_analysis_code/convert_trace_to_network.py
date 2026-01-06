import xml.etree.ElementTree as ET
import gzip
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

from tree_projection import project_to_plane_from_3d_tree
from get_features import coarsen_graph, total_edge_length, convex_hull_area, compute_void

def get_closest_node(G, n_coords):

	all_pts = nx.get_node_attributes(G, 'coords')

	dist_dict = {k: np.sqrt(np.sum((all_pts[k] - n_coords)**2)) for k in all_pts.keys()}

	sorted_by_dist = dict(sorted(dist_dict.items(), key=lambda item: item[1]))
	#closest = sorted_by_dist[0]

	#print(list(sorted_by_dist.keys())[:5])
	return list(sorted_by_dist.keys())[0]


def get_edge_length(G, n, m):

	path = nx.shortest_path(G, n, m)

	#print(path)

	tot_len = 0

	for i in range(len(path))[1:]:

		tot_len += np.sqrt(np.sum((G.nodes[path[i-1]]['coords'] -
								G.nodes[path[i]]['coords']) ** 2))

	#print(tot_len)
	return tot_len

def trace_file_to_G(filename):

	input = gzip.open(filename, 'r')
	tree = ET.parse(input)
	root = tree.getroot()

	special_xs = []
	special_ys = []
	special_zs = []

	G = nx.Graph()
	node_count = 1

	for path in root:

		if path.tag == 'path':
			#print(path.tag, path.attrib.keys())

			pathDict = path.attrib

			path_dict_keys = list(pathDict.keys())

			id = int(pathDict[path_dict_keys[0]])

			# ['id', 'swctype', 'color', 'channel', 'frame', 'spines', 'usefitted', 'fitted',
			# 	'startson', 'startx', 'starty',
			#	 'startz', 'startsindex', 'name', 'reallength'])

			fit_key = path_dict_keys.index('usefitted')


			if pathDict[path_dict_keys[fit_key]] == 'false':

				#print('now on path id', pathDict[path_dict_keys[0]])

				node_list = []

				first_pt = True

				for n in path.iter(tag='point'):

					nDict = n.attrib
					keys = list(nDict.keys())

					#print(nDict)

					xd = np.round(float(nDict[keys[3]]), 4)
					yd = np.round(float(nDict[keys[4]]), 4)
					zd = np.round(float(nDict[keys[5]]), 4)
					r = float(nDict[keys[0]])

					# if first node and node not 1, add edge to the closest node in existing network

					if first_pt and node_count > 1:
						special_xs.append(xd)
						special_ys.append(yd)
						special_zs.append(zd)
						first_pt = False

						#print('adding edge', node_count, get_closest_node(G, np.array((xd, yd, zd))))
						neighbor_node = get_closest_node(G, np.array((xd, yd, zd)))
						#G.add_edge(node_count, get_closest_node(G, np.array((xd, yd, zd))))

					# else: add edge to the previous node

					elif node_count > 1:
						neighbor_node = node_count-1
						#G.add_edge(node_count, node_count-1)

					G.add_node(node_count)
					G.nodes[node_count]['coords'] = np.array((xd, yd, zd))
					G.nodes[node_count]['radius'] = r

					if node_count > 1:
						G.add_edge(node_count, neighbor_node)

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


	# mapping = {}
	#
	# count = 1
	#
	# for n in G.nodes():
	# 	if G.degree(n) != 2:
	# 		mapping[n] = count
	# 		mapping[count] = n
	# 		count += 1
	# 	else:
	# 		mapping[n] = n
	#
	# G = nx.relabel_nodes(G, mapping)

	#print(G.number_of_nodes())

	return G

def graph_plotting(G, savename):

	fig, axes = plt.subplots(figsize=(5, 5))

	for n in G.nodes():
		c = G.nodes[n]['coords']
		plt.scatter(c[0], c[1], s=1, color = 'k')

		if len(list(G.neighbors(n))) == 3 or n == 1:
			plt.text(c[0], c[1], str(n), color = 'r', fontsize=6)


	plt.axis('equal')
	#plt.xlim(0, 700)
	plt.xlim(-50, 150)
	plt.ylim(-10, 200)
	plt.savefig(savename + '.pdf')


def process():

	info_dict = {}
	info_dict['number'] = []
	info_dict['instar'] = []
	info_dict['side'] = []
	info_dict['Tr'] = []
	info_dict['tree'] = []


	# instar labels:
		# L1: 1
		# L2: 2
		# Late L3: 3
		# Early L3: 4
	instar_folder_dict = {1: 'mCherry/Fixed_traces_mCherry_L1_1-30/',
						  2: 'mCherry/Fixed_traces_of_mCherry_L2_1-30/',
						  4: 'mCherry/Fixed_traces_of_mCherry_Early_L3/',
						  3: 'mCherry/BG2_x_mCherry_RNAi_Late_3d_Instar_(plate)/'}

	instars = [1, 2, 3, 4]
	#instars = [4]

	top_dir = 'TC_data/'

	for instar in instars:

		dir = top_dir + instar_folder_dict[instar]

		all_files = os.listdir(dir)

		for f in all_files:
			if f[len(f)-7:] == '.traces':

				if instar == 1:
					info = f.split('_')

					info2 = info[1].split(' - ')

					num = int(info2[0])
					metamere = info2[1][2]
					side = info2[2][0]

				if instar == 3:
					info = f.split('_')

					info2 = info[1].split(' - ')

					num = int(info2[0])
					info3 = info2[1].split(' ')

					metamere = info3[1][-1]
					side = info3[2][0]

				if instar == 2 or instar == 4:
					info = f.split('_')

					num = int(info[0])
					metamere = info[1][2]
					side = info[1][3]

				print(instar, num, metamere, side)

				G_3d = trace_file_to_G(dir + f)

				info_dict['number'].append(num)
				info_dict['instar'].append(instar)
				info_dict['side'].append(side)
				info_dict['Tr'].append(metamere)
				info_dict['tree'].append(G_3d)

				# graph_plotting(G_3d, str(num) + '_Tr' + str(metamere) + '_' + str(side),
				# 			   top_dir + 'L' + str(instar) + '/')

	dataframe = pd.DataFrame(info_dict, columns = ['number', 'instar', 'side', 'Tr', 'tree'])
	pickle.dump(dataframe, open(top_dir + 'retraced_mCherry_TCs_raw.p', "wb"))


def process_hybrid():
	info_dict = {}
	info_dict['number'] = []
	info_dict['instar'] = []
	info_dict['side'] = []
	info_dict['Tr'] = []
	info_dict['tree'] = []

	# instar labels:
	# L1: 1
	# L2: 2
	# Late L3: 3
	# Early L3: 4
	instar_folder_dict = {1: 'Fixed_traces_mCherry_L1_1-30/',
						  2: 'Fixed_traces_of_mCherry_L2_1-30/',
						  4: 'Fixed_traces_of_mCherry_Early_L3/',
						  3: '220611_All_Traces_Tanner_mCherryRNAi_LateL3/'}

	instars = [1, 2, 3, 4]
	#instars = [4]

	top_dir = 'TC_data/hybrid_data/'

	for instar in instars:

		dir = top_dir + instar_folder_dict[instar]

		all_files = os.listdir(dir)

		for f in all_files:
			if f[len(f) - 7:] == '.traces':

				print('now on instar:', instar, 'filename:', f)

				if instar == 1:
					info = f.split('_')
					info2 = info[1].split(' - ')

					num = int(info2[0])
					metamere = info2[1][2]
					side = info2[2][0]

				else:
					info = f.split('_')

					num = int(info[0])
					metamere = info[1][2]
					side = info[1][3]

				print(instar, num, metamere, side)

				G_3d = trace_file_to_G(dir + f)

				info_dict['number'].append(num)
				info_dict['instar'].append(instar)
				info_dict['side'].append(side)
				info_dict['Tr'].append(metamere)
				info_dict['tree'].append(G_3d)

		# graph_plotting(G_3d, str(num) + '_Tr' + str(metamere) + '_' + str(side),
		# 			   top_dir + 'L' + str(instar) + '/')

	dataframe = pd.DataFrame(info_dict, columns=['number', 'instar', 'side', 'Tr', 'tree'])
	pickle.dump(dataframe, open(top_dir + 'hybrid_mCherry_TCs_raw.p', "wb"))

#process_hybrid()

def process_time_lapse():

	info_dict = {}
	info_dict['number'] = []
	info_dict['instar'] = []
	info_dict['side'] = []
	info_dict['Tr'] = []
	info_dict['tree'] = []

	files = ['Larva1_tr8L_Day_1', 'Larva1_tr8L_Day_2']


	for f in files:

		G = trace_file_to_G(f + '.traces')

		# info_dict['number'].append(num)
		# info_dict['instar'].append(instar)
		# info_dict['side'].append(side)
		# info_dict['Tr'].append(metamere)
		# info_dict['tree'].append(G_3d)

		# fig, axes = plt.subplots(figsize=(5, 5))
		#
		# for n in G.nodes():
		# 	c = G.nodes[n]['coords']
		# 	plt.scatter(c[0], c[1], s=1, color='k')
		#
		# plt.axis('equal')
		# plt.savefig('internal_growth_fig/' + f + '_trace.pdf')

		H = coarsen_graph(G)

		for e in H.edges():
			c1 = np.array(H.nodes[e[0]]['coords'][:2])
			c2 = np.array(H.nodes[e[1]]['coords'][:2])
			H[e[0]][e[1]]['length'] = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

		print(G.number_of_edges(), H.number_of_edges())

		# print(total_edge_length(G), total_edge_length(H))
		# print(convex_hull_area(G), convex_hull_area(H))
		# print(compute_void(G), compute_void(H))

		print('A, L:', round(convex_hull_area(G)), round(total_edge_length(G)))

		coords = np.array(list(nx.get_node_attributes(G, 'coords').values()))
		maxs = coords.max(axis=0)
		mins = coords.min(axis=0)

		xspan = maxs[0] - mins[0]
		yspan = maxs[1] - mins[1]

		print('spans:', xspan, yspan)



	# dataframe = pd.DataFrame(info_dict, columns=['number', 'instar', 'side', 'Tr', 'tree'])
	# pickle.dump(dataframe, open(top_dir + 'hybrid_mCherry_TCs_raw.p', "wb"))

# process_time_lapse()