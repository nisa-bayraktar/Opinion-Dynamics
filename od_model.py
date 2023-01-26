from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

G = nx.DiGraph(nx.complete_graph(3))

#params
# c = float
# h = float
# a = float
# theta_h = float
# theta_a = float
# n = int
# x = float
# j = []
#n -> nodes 
#w -> weights 

#helper functions

def set_opinion_state(node,x):
     G.nodes[node]["opinion_state"] = x

def get_opinion_state(node):
    opinion_state = nx.get_node_attributes(G, "opinion_state")
    return opinion_state[node]

def set_weight(j,node,w):
    G[j][node]["weight"]= w

def update_opinion_state(conformity,node,epsilon):
    x_i = (conformity * (avg_neighbourhood(node) - get_opinion_state(node))) + epsilon
    return x_i

def update_weight(homophily,novelity,node,t_h,t_a):
    j = [n for n in G.neighbors(node)] #returns an array of the node's neighbors
    for i in range (len(j)):
        w_ij = (homophily * calculate_fh(t_h,node,G.nodes[j[i]]["opinion_state"])) + (novelity * calculate_fa(t_a,node,G.nodes[j[i]]["opinion_state"]))
        set_weight(j[i],node,w_ij)

def calculate_fh(t_h,node,xj):
    fh = t_h - abs(node - xj)
    return fh

def calculate_fa(t_a,node,xj):
    fa = abs(avg_neighbourhood(node)-xj) - t_a
    return fa

def avg_neighbourhood(node):
    s1_arr = []
    s2_arr = []
    j = [n for n in G.neighbors(node)] #returns an array of the node's neighbors
    for i in range (len(j)):
        j_s = G.nodes[j[i]]["opinion_state"]
        w_ij = G[j[i]][node]["weight"] # weight from node j(adj) to node i(source)
        m = (w_ij * j_s)
        s1_arr.append(m)
        s2_arr.append(w_ij)

    s1 = sum(s1_arr)
    s2 = sum(s2_arr)
    avg = s1 / s2
    return avg

def main():
    
    for n in list(G):
        x_s = np.random.standard_normal() #opinion state
        set_opinion_state(n,x_s)

    for n_i,n_j in G.edges():
        w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights
        set_weight(n_j,n_i,w_n)
       
    # set_weight(0,1,3)
    # set_weight(1,0,1)
 
    # print(G[1][0]["weight"])
    # print(G[2][0]["weight"])
    # print(G.nodes(data=True))
    # print(avg_neighbourhood(0))
    for t in range (0,100):
        e = np.random.normal(0,0.1)
        for n in list(G):
            update_opinion_state(0.3,n,e)
            update_weight(0.5,0.1,n,0.2,0.4)
        
        
           
    #print(get_opinion_state(0))  

if __name__ == "__main__":
    main()
pos = nx.spring_layout(G)

node_labels= nx.get_node_attributes(G,"opinion_state")
edge_lables= nx.get_edge_attributes(G,"weight")

nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_lables)
nx.draw_networkx_labels(G,pos,labels=node_labels)
nx.draw_networkx(G)
# plt.margins(0.01)
plt.show()
