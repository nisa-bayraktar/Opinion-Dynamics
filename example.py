from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

G = nx.MultiDiGraph([(5,1),(4,2),(3,4)])


print([n for n in G.neighbors(5)])
