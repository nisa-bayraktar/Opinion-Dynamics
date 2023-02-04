from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# import louvain-community ?? 

dt = 0.1
conformity = 0.01
novelty = 0.01
homophily = 0.3
t_h = 0.01
t_a = 0.03


G = nx.DiGraph(nx.complete_graph(100))



def update_opinion_state(node,epsilon):
    opinion_state = nx.get_node_attributes(G, "opinion_state")
    x_i = opinion_state[node]
    x_i += (conformity * (avg_neighbourhood(node) - x_i)) * dt
    x_i += epsilon
    G.nodes[node]["opinion_state"] = x_i
   

def update_weight(node):
    for i in G.neighbors(node):
        w_ij = (homophily * (t_h - abs(G.nodes[node]["opinion_state"] - G.nodes[i]["opinion_state"]))) + (novelty * (abs(avg_neighbourhood(node)- G.nodes[i]["opinion_state"]) - t_a))  * dt
        G[i][node]["weight"] += w_ij


def avg_neighbourhood(node):
    sum1 = 0
    sum2 = 0
    for j in G.neighbors(node):
        sum1 += G.nodes[j]["opinion_state"] * G[j][node]["weight"]
        sum2 += G[j][node]["weight"]

    avg = sum1 / sum2 # what happens ıf node has no neıghbours (all weıghts ınto node are equal to zero)?
    return avg


def main():

    for n in list(G):
        x_s = np.random.standard_normal() #opinion state
        G.nodes[n]["opinion_state"] = x_s

    for n_i,n_j in G.edges():
        w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights
        G[n_j][n_i]["weight"]= w_n
   
 
    for t in range (0,100): 
        
        for n in list(G):
            e = np.random.normal(0,0.1)
            update_opinion_state(n,e)
            update_weight(node=n)
        if t % 10 == 0:
            visualize_graph(G)
            # input("hit enter to continue")  
        print(t)      
  


def visualize_graph(graph):
    node_colours = [graph.nodes[i]["opinion_state"] for i in list(G)]
    nx.draw_networkx(graph,node_color=node_colours)
    plt.draw()
    plt.show()
    # input("hit enter to continue")
    # plt.close()

if __name__ == "__main__":
    main()

