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
# t_h = 0
# t_a = 0
# values = [0.01,0.03,0.1,0.3]

# conformity = 0.01 
# homophily = 0.3
# novelty = 0.01
# t_h = 0.01
# t_a = 0.03

conformity = 0.3
homophily = 0.01
novelty = 0.3
t_h = 0.3
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
        new_weights = []

        for node in list(G):
            epsilon = np.random.normal(0,0.1)

            if list(G.neighbors(node)) != []: # what happens ıf node has no neıghbours (all weıghts ınto node are equal to zero)?
                sum1 = 0
                sum2 = 0
                for j in G.neighbors(node):
                    sum1 += G.nodes[j]["opinion_state"] * G[node][j]["weight"]
                    sum2 += G[node][j]["weight"]
                x_i = G.nodes[node]["opinion_state"]
                if sum2>0:
                    avg = sum1 / sum2 
                    #calculate the node's new opinion state
                    x_i += (conformity * (avg - x_i))  * dt
                x_i += epsilon
                new_opinion_state.append(x_i)
          
            #update the node's weight from its neighbourhood
            for j in G.neighbors(node):
                w_ij = G[node][j]["weight"]
                w_ij += (homophily * (t_h - abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"])))   * dt
                if sum2>0:
                    w_ij += (novelty * (abs(avg - G.nodes[j]["opinion_state"]) - t_a)) * dt
                new_weights.append((node,j,w_ij))
        
        if list(G.neighbors(node)) != []:
            #update the node's opinion state
            for node in list(G):
                if sum2>0:

                    G.nodes[node]["opinion_state"] = new_opinion_state[node]

            for node, j, weight in new_weights:
            
                G.edges[node, j]["weight"] = weight

                if G[node][j]["weight"] < 0:
                    G[node][j]["weight"] =0

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


def main_asynch():


    #random initializations
    for n in G.nodes():
        x_s = np.random.standard_normal() #opinion state
        G.nodes[n]["opinion_state"] = x_s

    for n_i,n_j in G.edges():
        #print('setting weighjt', n_i, n_j)
        w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights - WHAT ABOUT WEIGHT (5,5)?
        G[n_i][n_j]["weight"]= w_n

    sum_opinion_state = 0
    std_dev = 0
    average_opinion_states = []
   
    max_edge_weight = 0
    min_edge_weight = 0

   
    for t in range (0,101): 

        nodes = list(G.nodes())
        
        np.random.shuffle(nodes)

        for node in nodes:
            neighbours = list(G.neighbors(node))
            
            
            # calculate the average neighbourhood of the node
            if len(neighbours)>0: # what happens ıf node has no neıghbours (all weıghts ınto node are equal to zero)?
                sum2 = sum(G[j][node]["weight"]for j in neighbours )
            
                if sum2>0:
                    avg = sum(G[j][node]["weight"] * G.nodes[j]['opinion_state'] for j in neighbours) / sum2
        
                    #calculate the node's new opinion state
                    G.nodes[node]["opinion_state"] += conformity * (avg - G.nodes[node]["opinion_state"])  * dt
        
                #update the node's weight from its neighbourhood
                for j in neighbours:
                    diff = abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"])
                    G[j][node]["weight"] += homophily * (t_h - diff)  * dt
                    if sum2>0:
                        diff = abs(avg - G.nodes[j]["opinion_state"])
                        G[j][node]["weight"] += novelty * (diff - t_a) * dt

            
                    if G[j][node]["weight"] < 0:
                        G[j][node]["weight"] =0
            G.nodes[node]["opinion_state"] += np.random.normal(0,0.1)
        

            

            

    UG = G.to_undirected()

    for node in G.nodes:
        for neighbour in G.neighbors(node):
    
            UG[neighbour][node]["weight"] = (
                (G[node][neighbour]["weight"] + G[neighbour][node]["weight"])/2
            )
            # if G[node][neighbour]["weight"] > max_edge_weight:
            #     max_edge_weight = G[node][neighbour]["weight"] 
            # if G[node][neighbour]["weight"]  < min_edge_weight:
            #     min_edge_weight = G[node][neighbour]["weight"] 
    

    #visualize_graph(UG)

    #calculate the average weight of all edges
    # s = 0
    # max_edge_weight = 0
    # min_edge_weight = 0
    
    s2 = sum(UG[node] [neighbour]["weight"] for node, neighbour in list(UG.edges))
    avg_weight = s2 / len(list(UG.edges))
   
    if avg_weight > max_edge_weight:
        max_edge_weight = avg_weight
    if avg_weight < min_edge_weight:
        min_edge_weight = avg_weight
    

    #find the communities of the graph
    a=nx_comm.louvain_communities(UG)

    #number of communities
    n_communities = len(a)
        
    #calculate the modularity of the community
    modularity = nx_comm.modularity(UG,a)
    #print (modularity)
    # sum_opinion_state = 0
    # std_dev = 0
    # average_opinion_states = []
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

    print(min_edge_weight,max_edge_weight)

                
        # if (t+1) % 50 == 0:
        #         visualize_graph(G)
# print(t)
        
        
       
       
            # UG = G.to_undirected()
            # for node in list(G):
            #     for neighbour in G.neighbors(node):
            #             UG.edges[node, neighbour]["weight"] = (
            #                 (G.edges[node, neighbour]["weight"] + G.edges[neighbour, node]["weight"])/2
            #             )
            #     #visualize_graph(UG)

            # #calculate the average weight of all edges
            # s = sum(UG.edges[node, neighbour]["weight"] for node, neighbour in UG.edges)
            # avg_weight = s / len(UG.edges)
            # #print(avg_weight)
            
            # #find the communities of the graph
            # a=nx_comm.louvain_communities(UG)

            # #number of communities
            # n_communities = len(a)
                
            # #calculate the modularity of the community
            # modularity = nx_comm.modularity(UG,a)
            # #print (modularity)

            # sum_opinion_state = 0
            # std_dev = 0
            # average_opinion_states = []
            # for i in range (len(a)):
            #     for k in a[i]:
            #         #calculate the average opinion state of communities
            #         sum_opinion_state += UG.nodes[k]["opinion_state"]
            #     average_opinion_state = sum_opinion_state / len(a[i])
            #     average_opinion_states.append(average_opinion_state)

            #     #calculate the standard deviation of community states
            #     std_dev += (UG.nodes[k]["opinion_state"] - average_opinion_state)**2
            #     std_dev = std_dev / len(a[i])
                    
            # #calcuate the range of opinion states
            # range_community = max(average_opinion_states) - min(average_opinion_states)
            #     #print(range_community)
                

            # dict_data = {'h_values':homophily,'n_values':novelty,'th_values':t_h,'ta_values':t_a,'c_values': conformity,'avg_weight': avg_weight, 'std_dev':std_dev, 'average_os': average_opinion_state, 'range_community': range_community, 'n_communities': n_communities,'modularity': modularity}
            # file_exists = os.path.isfile('output2.csv')

            # with open('output2.csv', mode='a') as csv_file:
                
            #         fieldnames = ['h_values','n_values','th_values','ta_values','c_values','avg_weight', 'std_dev', 'average_os', 'range_community', 'n_communities','modularity']
            #         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            #         if not file_exists:
            #             writer.writeheader() 
            #         writer.writerow(dict_data)
                    
            #         csv_file.close()
    #print(min_edge_weight,max_edge_weight)

def visualize_graph(graph):
    edge_colours = [graph[u][v]['weight'] for u,v in graph.edges()]
    node_colours = [graph.nodes[i]["opinion_state"] for i in list(graph)]
    nx.draw_networkx(graph,node_color=node_colours,cmap='rainbow', edge_cmap =plt.cm.Greys,edge_color=edge_colours)
    plt.draw()
    plt.show()
    # input("hit enter to continue")
    # plt.close()
def test():
   
    lst = []
    for i in range(100):
        lst.append(i)

if __name__ == "__main__":
    main_asynch()
    #visualize_graph(G)

    # For Python>=3.5 one can also write:
    #print(timeit.timeit("test()", globals=locals()))

