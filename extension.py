from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import networkx.algorithms.community as nx_comm
import csv
import os.path
import timeit
import itertools
import pickle as pickle


G = pickle.load(open('new_choice_network_0.03/new_1000_network_choice_10', 'rb'))

UG = G.to_undirected()
for node in list(G.nodes()):

    for neighbour in G.neighbors(node):

        UG.edges[node, neighbour]["weight"] = (
                (G.edges[node, neighbour]["weight"] + G.edges[neighbour, node]["weight"])/2
            )
   
avg_op_node = sum(UG.nodes[node]["opinion_state"]for node in list(UG.nodes)) 
avg_op = avg_op_node / len(list(UG.nodes))
nodes = list(UG.nodes)

for node in nodes:
    neighbours = list(UG.neighbors(node))
    global_ecc = abs(UG.nodes[node]["opinion_state"] - avg_op)

    # two idea: global eccentricity .. abs difference between node opinion and average network opinion
    UG.nodes[node]["global_eccentricity"] = global_ecc
   
       
#find the communities of the graph
a=nx_comm.louvain_communities(UG)

  
for i in range (len(a)):
    sum_opinion_state = 0
    for node in a[i]:
        sum_opinion_state += UG.nodes[node]["opinion_state"]
        com_size = len(a[i])
        UG.nodes[node]["community_size"]=com_size
    average_opinion_state = sum_opinion_state / len(a[i])
    com_ecc= abs(average_opinion_state - avg_op)
    
    for n in a[i]:
       
        UG.nodes[n]["community_eccentricity"]= com_ecc
        UG.nodes[n]["within_community_eccentricity"]= abs(UG.nodes[n]["opinion_state"] - average_opinion_state)


for n in range (len(UG.nodes)):
        dict_data = {'h_values': UG.nodes[n]["h"],'a_values': UG.nodes[n]["a"],'c_values':  UG.nodes[n]["c"],'global_node_eccentricity': UG.nodes[n]["global_eccentricity"],'within_community_eccentricity': UG.nodes[n]["within_community_eccentricity"],'community_eccentricity': UG.nodes[n]["community_eccentricity"],'community_size':UG.nodes[n]["community_size"]}
        file_exists = os.path.isfile('new_1000_network_choice_0.03.csv')

        with open('new_1000_network_choice_0.03.csv', mode='a') as csv_file:
            
                fieldnames = ['h_values','a_values','c_values','global_node_eccentricity','within_community_eccentricity','community_eccentricity','community_size']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader() 
                writer.writerow(dict_data)
            
        csv_file.close()
