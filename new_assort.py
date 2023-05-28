import numpy as np
import pickle
import networkx as nx
import csv

network_files = ['new_normal_network_0.03/new_1000_network_normal_1', 'new_normal_network_0.03/new_1000_network_normal_2', 'new_normal_network_0.03/new_1000_network_normal_3','new_normal_network_0.03/new_1000_network_normal_4','new_normal_network_0.03/new_1000_network_normal_5','new_normal_network_0.03/new_1000_network_normal_6','new_normal_network_0.03/new_1000_network_normal_7','new_normal_network_0.03/new_1000_network_normal_8','new_normal_network_0.03/new_1000_network_normal_9','new_normal_network_0.03/new_1000_network_normal_10']
networks = []
for network_file in network_files:
    with open(network_file, 'rb') as f:
        network = pickle.load(f)
        networks.append(network)

# Combine the networks into a single graph
G = nx.compose_all(networks)

attributes = ['c', 'h', 'a']
combinations = [(X, Y) for X in attributes for Y in attributes]

dict_data = {}
with open('NEW_normal_assortativity.csv', mode='w') as csv_file:
    fieldnames = ['XY', 'assortativity']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()  # write the header row
    for X, Y in combinations:
        W = 0
        Xsou = 0
        Ytar = 0
        enum = 0
        sigmaX = 0
        sigmaY = 0
        for u,v,d in G.edges(data=True):
            W += d['weight']

        for i,j in G.edges():
            Xsou +=(G.nodes[i][X] * G[i][j]['weight']) 
            Ytar += (G.nodes[j][Y] * G[j][i]['weight']) 
        Xsou = Xsou / W
        Ytar = Ytar / W
        for i,j in G.edges():
            sigmaX += (G.nodes[i][X] - Xsou) ** 2 * G[i][j]['weight']
            sigmaY += (G.nodes[j][Y] - Ytar) ** 2 * G[j][i]['weight']
            enum += (G.nodes[i][X] - Xsou) * (G.nodes[j][Y] - Ytar) * G[i][j]['weight']

        sigma_X_square = np.sqrt(sigmaX / W)
        sigma_Y_square = np.sqrt(sigmaY / W)

        denum = (W * sigma_X_square  * sigma_Y_square)
        assortativity = enum / denum

        dict_data['XY'] = f'{X}{Y}'
        dict_data['assortativity'] = assortativity
        writer.writerow(dict_data) 
