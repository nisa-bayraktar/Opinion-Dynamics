from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import networkx.algorithms.community as nx_comm

dt = 0.1
conformity = 0.01 #0.3 
homophily = 0.3 #0.01
novelty = 0.01 #0.3
t_h = 0.01 #0.3
t_a = 0.03
new_opinion_state = []


G = nx.DiGraph(nx.complete_graph(100))


def main():

    #random initializations
    for n in G.nodes():
        x_s = np.random.standard_normal() #opinion state
        G.nodes[n]["opinion_state"] = x_s

    for n_i,n_j in G.edges():
        w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights
        G[n_j][n_i]["weight"]= w_n
    
    # t iterations
    for t in range (0,100): 
        
        for node in G.nodes():
            epsilon = np.random.normal(0,0.1)

            # calculate the average neighbourhood of the node
            if G.neighbors(node) == []: # what happens ıf node has no neıghbours (all weıghts ınto node are equal to zero)?
                avg = 0
            
            else:
                sum1 = 0
                sum2 = 0
                for j in G.neighbors(node):
                    sum1 += G.nodes[j]["opinion_state"] * G[j][node]["weight"]
                    sum2 += G[j][node]["weight"]
                avg = sum1 / sum2 

            #calculate the node's new opinion state
            x_i = G.nodes[node]["opinion_state"]
            x_i += (conformity * (avg - x_i)) * dt
            x_i += epsilon
            new_opinion_state.append(x_i)

            #update the node's weight from its neighbourhood
            for j in G.neighbors(node):
                w_ij = G[j][node]["weight"]
                w_ij += ((homophily * (t_h - abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"]))) + (novelty * (abs(avg - G.nodes[j]["opinion_state"]) - t_a)))  * dt
                G[j][node]["weight"] = w_ij
            
        #update the node's opinion state
        for node in G.nodes():
            G.nodes[node]["opinion_state"] = new_opinion_state[node]
                          
        # if t % 10 == 0:
        #      visualize_graph(G)
        # if t == 99:
        #     UG = G.to_undirected()
        #     for node in G.nodes():
        #         for neighbour in G.neighbors(node):
        #                 UG.edges[node, neighbour]["weight"] = (
        #                     (G.edges[node, neighbour]["weight"] + G.edges[neighbour, node]["weight"])/2
        #                 )
        #     visualize_graph(UG)
    a=nx_comm.louvain_communities(G)
    print(a)
    print(len(a))
                        
            
            # input("hit enter to continue")  
            
        


def visualize_graph(graph):
    node_colours = [graph.nodes[i]["opinion_state"] for i in G.nodes()]
    nx.draw_networkx(graph,node_color=node_colours)
    plt.draw()
    plt.show()
    # input("hit enter to continue")
    # plt.close()

if __name__ == "__main__":
    main()

