from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

G = nx.MultiDiGraph()
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
w = np.random.uniform([0,1])
for n in range (1,100):
    G.add_edges_from(n,G.neighbors(n))

def main():
    return

#helper functions

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
    j = G.neighbors(node)
    w_ij = G[node][j]["weight"]
    s1 = sum(w_ij * j)
    s2 = sum(w_ij)
    avg = s1 / s2
    return avg

#nx.draw(G)
nx.draw_networkx(G)
plt.margins(0.01)
plt.show()