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
# conformity = 0 
# homophily = 0
# novelty = 0
t_h = 0.01
t_a = 0.01
#values = [0.01,0.03,0.1,0.3]



def main_asynch():

    G = nx.DiGraph(nx.complete_graph(100))

        #random initializations
    for n in list(G.nodes()):
        x_s = np.random.standard_normal() #opinion state
        G.nodes[n]["opinion_state"] = x_s

        G.nodes[n]["c"] = np.random.choice([0.01, 0.03, 0.1, 0.3])
        G.nodes[n]["h"] = np.random.choice([0.01, 0.03, 0.1, 0.3])
        G.nodes[n]["a"] = np.random.choice([0.01, 0.03, 0.1, 0.3])
        #print (G.nodes[n]["a"])


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
                    G.nodes[node]["opinion_state"] += G.nodes[node]["h"] * (avg - G.nodes[node]["opinion_state"])  * dt
        
                #update the node's weight from its neighbourhood
                for j in neighbours:
                    dif = abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"])
                    G[node][j]["weight"] += G.nodes[n]["c"] * (t_h - dif)  * dt
                    if sum2>0:
                        dif = abs(avg - G.nodes[j]["opinion_state"]) 
                        G[node][j]["weight"] +=  G.nodes[n]["a"] * (dif - t_a) * dt

                    if G[node][j]['weight'] < 0:
                        G[node][j]['weight'] = 0



            G.nodes[node]["opinion_state"] += epsilon
    
    

    UG = G.to_undirected()
    for node in list(G.nodes()):
        for neighbour in G.neighbors(node):

            UG.edges[node, neighbour]["weight"] = (
                    (G.edges[node, neighbour]["weight"] + G.edges[neighbour, node]["weight"])/2
                )
        print(node,":",UG.nodes[node]["a"])

    
    visualize_graph(UG)


def visualize_graph(graph):
    edge_colours = [graph[u][v]['weight'] for u,v in graph.edges()]
    node_colours = [graph.nodes[i]["opinion_state"] for i in list(graph)]
    nx.draw_networkx(graph,node_color=node_colours,cmap='rainbow', edge_cmap =plt.cm.Greys,edge_color=edge_colours)
    plt.draw()
    plt.show()


if __name__ == "__main__":
    main_asynch()
   
  

