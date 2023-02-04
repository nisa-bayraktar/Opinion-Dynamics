from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import networkx.algorithms.community as nx_comm
import csv
import os.path

  

dt = 0.1
conformity = 0.01 #0.3 #  
homophily = 0.3 #0.01 # 
novelty = 0.01 #0.3 # 
t_h = 0.01 #0.3 #
t_a = 0.03

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
    
    # t iterations
    for t in range (0,100): 
        
        new_opinion_state = []
        new_weıghts = []

        for node in list(G):
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
            x_i += (conformity * (avg - x_i))  * dt
            x_i += epsilon
            new_opinion_state.append(x_i)
          
            #update the node's weight from its neighbourhood
            for j in G.neighbors(node):
                w_ij = G[j][node]["weight"]
                w_ij += ((homophily * (t_h - abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"]))) + (novelty * (abs(avg - G.nodes[j]["opinion_state"]) - t_a)))  * dt
                new_weıghts.append((node,j,w_ij))

        
        #update the node's opinion state
        for node in list(G):
            G.nodes[node]["opinion_state"] = new_opinion_state[node]

        for i, j, weıght in new_weıghts:
            G.edges[i, j]["weight"] = weıght

        # if t % 100 == 0:
        #      visualize_graph(G)
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
        

    dict_data = {'c_values': conformity,'avg_weight': avg_weight, 'std_dev':std_dev, 'average_os': average_opinion_state, 'range_community': range_community, 'n_communities': n_communities,'modularity': modularity}
    file_exists = os.path.isfile('output.csv')

    with open('output.csv', mode='a') as csv_file:
        
            fieldnames = ['c_values','avg_weight', 'std_dev', 'average_os', 'range_community', 'n_communities','modularity']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader() 
            writer.writerow(dict_data)
            
            csv_file.close()


def visualize_graph(graph):
    edge_colours = [graph[u][v]['weight'] for u,v in graph.edges()]
    node_colours = [graph.nodes[i]["opinion_state"] for i in list(graph)]
    nx.draw_networkx(graph,node_color=node_colours,edge_cmap =plt.cm.Greys,edge_color=edge_colours)
    plt.draw()
    plt.show()
    # input("hit enter to continue")
    # plt.close()


def main_asynch():

    global G

    #random initializations
    for n in G.nodes():
        x_s = np.random.standard_normal() #opinion state
        G.nodes[n]["opinion_state"] = x_s

    for n_i,n_j in G.edges():
        #print('setting weighjt', n_i, n_j)
        w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights - WHAT ABOUT WEIGHT (5,5)?
        G[n_j][n_i]["weight"]= w_n
        print(w_n)
    
    # t iterations
    for t in range (0,100): 

        print(t)
        nodes = list(G)
        np.random.shuffle(nodes)
        print(nodes)

        for node in nodes:
            epsilon = np.random.normal(0,0.1)

            # calculate the average neighbourhood of the node
            if list(G.neighbors(node)): # what happens ıf node has no neıghbours (all weıghts ınto node are equal to zero)?
                sum1 = 0
                sum2 = 0
                for j in G.neighbors(node):
                    sum1 += G.nodes[j]["opinion_state"] * G[node][j]["weight"]
                    sum2 += G[node][j]["weight"]
                if sum2>0:
                    avg = sum1 / sum2 
           
                    #calculate the node's new opinion state
                    G.nodes[node]["opinion_state"] += (conformity * (avg - G.nodes[node]["opinion_state"]))  * dt
          
            #update the node's weight from its neighbourhood
            for j in G.neighbors(node):
                G[node][j]["weight"] += ((homophily * (t_h - abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"]))) + \
                                         (novelty * (abs(avg - G.nodes[j]["opinion_state"]) - t_a)))  * dt

                
                if G[node][j]["weight"] < 0:
                    G[node][j]["weight"] =0

            G.nodes[node]["opinion_state"] += epsilon
                           
        if (t+1) % 50 == 0:
             visualize_graph(G)
        print(t)
        
        
       
       
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
        

    dict_data = {'c_values': conformity,'avg_weight': avg_weight, 'std_dev':std_dev, 'average_os': average_opinion_state, 'range_community': range_community, 'n_communities': n_communities,'modularity': modularity}
    file_exists = os.path.isfile('output.csv')

    with open('output.csv', mode='a') as csv_file:
        
            fieldnames = ['c_values','avg_weight', 'std_dev', 'average_os', 'range_community', 'n_communities','modularity']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader() 
            writer.writerow(dict_data)
            
            csv_file.close()


if __name__ == "__main__":
    main_asynch()
    visualize_graph(G)

