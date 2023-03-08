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


G = pickle.load(open('1000_network_start_uniform_5', 'rb'))


UG = G.to_undirected()
for node in list(G.nodes()):

    for neighbour in G.neighbors(node):

        UG.edges[node, neighbour]["weight"] = (
                (G.edges[node, neighbour]["weight"] + G.edges[neighbour, node]["weight"])/2
            )
   


# #calculate the average weight of all edges
# s = sum(UG.edges[node, neighbour]["weight"] for node, neighbour in UG.edges)
# avg_weight = s / len(list(UG.edges))
#print(avg_weight)
node_eccentricity = []
avg_op = sum(UG.nodes[node]["opinion_state"]for node in list(UG.nodes)) / len(list(UG.nodes))
nodes = list(G.nodes)
for node in nodes:
    neighbours = list(G.neighbors(node))
    sum_neighbours = sum(G.nodes[j]["opinion_state"]for j in neighbours)
    avg_n_os = sum_neighbours / len(neighbours)
    node_ecc = abs(G.nodes[node]["opinion_state"] - avg_n_os)
    node_eccentricity.append(node_ecc)
#print(node_eccentricity)
        

#find the communities of the graph
a=nx_comm.louvain_communities(UG)



#number of communities
n_communities = len(a)
    
#calculate the modularity of the community
modularity = nx_comm.modularity(UG,a)

community_ecc = []
community_size  = []
for node in (UG.nodes):
    for i in range (len(a)):
        sum_opinion_state = 0
        if node in a[i]:
            com_size = len(a[i])
            community_size.append(com_size)
            
            for k in a[i]:
                sum_opinion_state += UG.nodes[k]["opinion_state"]
            average_opinion_state = sum_opinion_state / len(a[i])
            com_ecc= abs(average_opinion_state - avg_op)
            community_ecc.append(com_ecc)
               

                        
          
     
# for i in range (len(a)):
#     sum_opinion_state = 0

#     for k in a[i]:
#         #calculate the average opinion state of communities
#         sum_opinion_state += UG.nodes[k]["opinion_state"]
    
#     average_opinion_state = sum_opinion_state / len(a[i])
#     com_ecc= abs(average_opinion_state - avg_op)
#     com_size = len(a[i])
#     comm_ecc.append(com_ecc)
#     community_size.append(com_size)
# print(comm_ecc,community_size)

    

# std_dev = np.std(average_opinion_states)
        
# #calcuate the range of opinion states
# range_community = max(average_opinion_states) - min(average_opinion_states)

for n in range (len(UG.nodes)):
        dict_data = {'h_values': UG.nodes[n]["h"],'a_values': UG.nodes[n]["a"],'c_values':  UG.nodes[n]["c"],'node_eccentricity': node_eccentricity[n],'community_opinion_eccentricity': community_ecc[n],'community_size':community_size[n]}
        file_exists = os.path.isfile('1000_network_start_uniform_5.csv')

        with open('1000_network_start_uniform_5.csv', mode='a') as csv_file:
            
                fieldnames = ['h_values','a_values','c_values','node_eccentricity','community_opinion_eccentricity','community_size']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader() 
                writer.writerow(dict_data)
            
        csv_file.close()
