from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# import louvain-community ?? 

dt = 0.1

G = nx.DiGraph(nx.complete_graph(10))

#helper functions

def set_opinion_state(node,x):
     G.nodes[node]["opinion_state"] = x

def get_opinion_state(node):
    opinion_state = nx.get_node_attributes(G, "opinion_state")
    return opinion_state[node]

def set_weight(j,node,w):
    G[j][node]["weight"]= w

def update_opinion_state(conformity,node,epsilon):
    x_i += (conformity * (avg_neighbourhood(node) - get_opinion_state(node))) * dt
    x_i += epsilon
    set_opinion_state(node,x_i)

def update_weight(homophily,novelty,node,t_h,t_a):
    j = [n for n in G.neighbors(node)] #returns an array of the node's neighbors
    for i in range (len(j)):
        w_ij = (homophily * calculate_fh(t_h,node,G.nodes[j[i]]["opinion_state"])) + (novelty * calculate_fa(t_a,node,G.nodes[j[i]]["opinion_state"])) # * dt
        set_weight(j[i],node,w_ij)
        # G[j][node]["weight"] += ...

def calculate_fh(t_h,node,xj):
    fh = t_h - abs(node - xj)
    return fh

def calculate_fa(t_a,node,xj):
    fa = abs(avg_neighbourhood(node)-xj) - t_a
    return fa

def avg_neighbourhood(node):
    s1_arr = []
    s2_arr = []
    sum1 = 0
    sum2 = 0
    j = [n for n in G.neighbors(node)] #returns an array of the node's neighbors
    for i in range (len(j)):
        j_s = G.nodes[j[i]]["opinion_state"]
        w_ij = G[j[i]][node]["weight"] # weight from node j(adj) to node i(source)
        m = (w_ij * j_s)
        s1_arr.append(m)
        s2_arr.append(w_ij)

    s1 = sum(s1_arr)
    s2 = sum(s2_arr)
    avg = s1 / s2 # what happens ıf node has no neıghbours (all weıghts ınto node are equal to zero)?
    return avg
# for i ın G.neıghbours(node):

def main():

    homophily = 0.5
    novelty = 0.1
    t_a = 0.2
    t_h = 0.4

    for n in list(G):
        x_s = np.random.standard_normal() #opinion state
        set_opinion_state(n,x_s)

    for n_i,n_j in G.edges():
        w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights
        set_weight(n_j,n_i,w_n)
   
    #print("start",G.nodes(data=True))
    for t in range (0,5): # is it how the time t works?
        # e = np.random.normal(0,0.1)
        print(t)
        for n in list(G):
            e = np.random.normal(0,0.1)
            update_opinion_state(0.3,n,e)
            update_weight(homophily=homophily,novelty=novelty,node=n,t_h=t_h,t_a=t_a)
            #print("in the loop",G.nodes(data=True))
    #print("end",G.nodes(data=True))
        visualize_graph(G)        
        input('hit reuturn')   
    #print(get_opinion_state(0))  

def visualize_graph(G):
    pos = nx.spring_layout(G)

    node_labels= nx.get_node_attributes(G,"opinion_state")
    edge_lables= nx.get_edge_attributes(G,"weight")
    node_colours = [G.nodes[i]["opinion_state"] for i in list(G)]
    # nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_lables)
    # nx.draw_networkx_labels(G,pos,labels=node_labels)
    nx.draw_networkx(G,node_color=node_colours)
    # plt.margins(0.01)
    plt.show()
    input('hit reuturn')   
    plt.close()

if __name__ == "__main__":
    main()


