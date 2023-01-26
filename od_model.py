from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

G = nx.DiGraph(nx.complete_graph(10))



#params
c = float
h = float
a = float
theta_h = float
theta_a = float
n = int
x = float
j = []
#n -> nodes 
#w -> weights 

#helper functions

def set_opinion_state(node,x):
     G.nodes[node]["opinion_state"] = x

def get_opinion_state(node):
    opinion_state = nx.get_node_attributes(G, "opinion_state")
    opinion_state[node]

def set_weight(node,j,w):
    G[node][j]['weight']= w

def update_opinion_state(conformity,node,epsilon):
    x_i = (conformity * (avg_neighbourhood(node) - node)) + epsilon
    return x_i

def update_weight(homophily,novelity,node,t_h,t_a,xj):
    w_ij = (homophily * calculate_fh(t_h,node,xj)) + (novelity * calculate_fa(t_a,node,xj))
    return w_ij

def calculate_fh(t_h,node,xj):
    fh = t_h - abs(node - xj)
    return fh

def calculate_fa(t_a,node,xj):
    fa = abs(avg_neighbourhood(node)-xj) - t_a
    return fa

def avg_neighbourhood(node):
    j = [n for n in G.neighbors(node)] #returns an array of the node's neighbors
    w_ij = G[node][j]["weight"]
    s1 = sum(w_ij * j)
    s2 = sum(w_ij)
    avg = s1 / s2
    return avg

def main():

    for t in range (0,100):
        w = np.random.uniform(0.0, np.nextafter(1,2))
        # G.add_edges_from(n,G.neighbors(n))
      


if __name__ == "__main__":
    main()


nx.draw_networkx(G)
plt.margins(0.01)
plt.show()
