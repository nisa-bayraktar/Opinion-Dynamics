from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import networkx.algorithms.community as nx_comm
import pickle as pickle
import csv
import os.path
import timeit
import itertools



dt = 0.1
t_h = 0.03
t_a = 0.03



def main_asynch():

    G = nx.DiGraph(nx.complete_graph(1000))


        #random initializations
    for n in list(G.nodes()):
        x_s = np.random.standard_normal() #opinion state
        G.nodes[n]["opinion_state"] = x_s

        # #random normal distribution
        # lower_bound = 0.01
        # upper_bound = 0.3
        # mean = (lower_bound + upper_bound) / 2
        # std_dev = (upper_bound - lower_bound) / 6
        # samples = []
        # for i in range(3):
        #     sample = np.random.normal(loc=mean, scale=std_dev, size=1)[0]
        #     while sample < lower_bound or sample > upper_bound:
        #         sample = np.random.normal(loc=mean, scale=std_dev, size=1)[0]
        #     samples.append(sample)

        # G.nodes[n]["c"] = samples[0]
        # G.nodes[n]["h"] = samples[1]
        # G.nodes[n]["a"] = samples[2]

        # #random choice
        # G.nodes[n]["c"] = np.random.choice([0.01, 0.03, 0.1, 0.3])
        # G.nodes[n]["h"] = np.random.choice([0.01, 0.03, 0.1, 0.3])
        # G.nodes[n]["a"] = np.random.choice([0.01, 0.03, 0.1, 0.3])
        
        #random uniform distribution
        random_c = np.random.uniform(0.01, 0.3 + 0.0001)
        random_h = np.random.uniform(0.01, 0.3 + 0.0001)
        random_a = np.random.uniform(0.01, 0.3 + 0.0001)

    
        G.nodes[n]["c"] = np.clip(random_c,0.01, 0.3)
        G.nodes[n]["h"] = np.clip(random_h,0.01, 0.3)
        G.nodes[n]["a"] = np.clip(random_a,0.01, 0.3)
        


    for n_i,n_j in G.edges():
        #print('setting weighjt', n_i, n_j)
        w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights - WHAT ABOUT WEIGHT (5,5)?
        G[n_i][n_j]["weight"]= w_n

    # t iterations
    for t in range (0,1000): 
        
        nodes = list(G)
        np.random.shuffle(nodes)

        for node in nodes:
            neighbours = list(G.neighbors(node))
            epsilon = np.random.normal(0,0.1)
            
            # calculate the average neighbourhood of the node
            if len(neighbours)>0: # what happens ıf node has no neıghbours (all weıghts ınto node are equal to zero)?
                
                
                sum2 = sum(G[node][j]["weight"]for j in neighbours)
                if sum2>0:
                    avg = sum(G.nodes[j]["opinion_state"] * G[node][j]["weight"] for j in neighbours) / sum2
        
                    #calculate the node's new opinion state
                    G.nodes[node]["opinion_state"] += G.nodes[n]["c"] * (avg - G.nodes[node]["opinion_state"])  * dt
        
                #update the node's weight from its neighbourhood
                for j in neighbours:
                    dif = abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"])
                    G[node][j]["weight"] += G.nodes[n]["h"] * (t_h - dif)  * dt
                    if sum2>0:
                        dif = abs(avg - G.nodes[j]["opinion_state"]) 
                        G[node][j]["weight"] +=  G.nodes[n]["a"] * (dif - t_a) * dt

                    if G[node][j]['weight'] < 0:
                        G[node][j]['weight'] = 0



            G.nodes[node]["opinion_state"] += epsilon

    
    pickle.dump(G, open('1000_network_uniform_6', 'wb'))


if __name__ == "__main__":
    main_asynch()


