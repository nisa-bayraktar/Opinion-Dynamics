from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import networkx.algorithms.community as nx_comm
import csv
import os.path
import timeit
import itertools


dt = 0.1
conformity = 0 
homophily = 0
novelty = 0
t_h = 0
t_a = 0
values = [0.01,0.03,0.1,0.3]

# conformity = 0.01 
# homophily = 0.3
# novelty = 0.01
# t_h = 0.01
# t_a = 0.03

# conformity = 0.3
# homophily = 0.01
# novelty = 0.3
# t_h = 0.3
# t_a = 0.03


G = nx.DiGraph(nx.complete_graph(100))

def main_synch():

    #random initializations
    for n in G.nodes():
        x_s = np.random.standard_normal() #opinion state
        G.nodes[n]["opinion_state"] = x_s

    for n_i,n_j in G.edges():
        #print('setting weighjt', n_i, n_j)
        w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights - WHAT ABOUT WEIGHT (5,5)?
        G[n_j][n_i]["weight"]= w_n

    parameters = [0, 0, 0, 0, 0]
    

    for combination in itertools.product(*[values for i in range(len(parameters))]):
            parameters = combination
        
            conformity = parameters[0]
            homophily = parameters[1]
            novelty = parameters[2]
            t_h = parameters[3]
            t_a = parameters[4]
            print(conformity)
      
    
            # t iterations
            for t in range (0,100): 
                
                new_opinion_state = []
                new_weights = []

                for node in list(G):
                    epsilon = np.random.normal(0,0.1)

                    if list(G.neighbors(node)) != []: 
                        sum1 = 0
                        sum2 = 0
                        for j in G.neighbors(node):
                            sum1 += G.nodes[j]["opinion_state"] * G[node][j]["weight"]
                            sum2 += G[node][j]["weight"]
                        x_i = G.nodes[node]["opinion_state"]
                        if sum2>0:
                            avg = sum1 / sum2 
                            #calculate the node's new opinion state
                            x_i += (conformity * (avg - x_i))  * dt
                        x_i += epsilon
                        new_opinion_state.append(x_i)
                
                        #update the node's weight from its neighbourhood
                        for j in G.neighbors(node):
                            w_ij = G[node][j]["weight"]
                            w_ij += (homophily * (t_h - abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"])))   * dt
                            if sum2>0:
                                w_ij += (novelty * (abs(avg - G.nodes[j]["opinion_state"]) - t_a)) * dt
                            new_weights.append((node,j,w_ij))

                if list(G.neighbors(node)) != []:

                
                    #update the node's opinion state
                    for node in list(G):
                        if sum2>0:

                            G.nodes[node]["opinion_state"] = new_opinion_state[node]

                    for node, j, weight in new_weights:
                    
                        G.edges[node, j]["weight"] = weight

                        if G[node][j]["weight"] < 0:
                            G[node][j]["weight"] =0

                # if (t+1) % 50 == 0:
                #     visualize_graph(G)
                # print(t)
                
                
            
            
            UG = G.to_undirected()
            for node in list(G):
                for neighbour in G.neighbors(node):
                        UG.edges[node, neighbour]["weight"] = (
                            (G.edges[node, neighbour]["weight"] + G.edges[neighbour, node]["weight"])/2
                        )
                #visualize_graph(UG)

            #calculate the average weight of all edges
            s = sum(UG.edges[node, neighbour]["weight"] for node, neighbour in UG.edges)
            avg_weight = s / len(UG.edges)
            #print(avg_weight)
            
            #find the communities of the graph
            a=nx_comm.louvain_communities(UG)

            #number of communities
            n_communities = len(a)
                
            #calculate the modularity of the community
            modularity = nx_comm.modularity(UG,a)
            #print (modularity)

            sum_opinion_state = 0
            std_dev = 0
            average_opinion_states = []
            for i in range (len(a)):
                for k in a[i]:
                    #calculate the average opinion state of communities
                    sum_opinion_state += UG.nodes[k]["opinion_state"]
                average_opinion_state = sum_opinion_state / len(a[i])
                average_opinion_states.append(average_opinion_state)

                #calculate the standard deviation of community states
                std_dev += (UG.nodes[k]["opinion_state"] - average_opinion_state)**2
                std_dev = std_dev / len(a[i])
                    
            #calcuate the range of opinion states
            range_community = max(average_opinion_states) - min(average_opinion_states)
                #print(range_community)
                

            dict_data = {'h_values':homophily,'n_values':novelty,'th_values':t_h,'ta_values':t_a,'c_values': conformity,'avg_weight': avg_weight, 'std_dev':std_dev, 'average_os': average_opinion_state, 'range_community': range_community, 'n_communities': n_communities,'modularity': modularity}
            file_exists = os.path.isfile('sync.csv')

            with open('sync.csv', mode='a') as csv_file:
                
                    fieldnames = ['h_values','n_values','th_values','ta_values','c_values','avg_weight', 'std_dev', 'average_os', 'range_community', 'n_communities','modularity']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader() 
                    writer.writerow(dict_data)
                    
                    csv_file.close()

