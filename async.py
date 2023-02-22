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
conformity = 0 
homophily = 0
novelty = 0
t_h = 0
t_a = 0
values = [0.01,0.03,0.1,0.3]



def main_asynch():

    G = nx.DiGraph(nx.complete_graph(1000))

    max_edge_weight = 0
    min_edge_weight = 0

    parameters = [0, 0, 0, 0, 0]


    for combination in itertools.product(*[values for i in range(len(parameters))]):

            #random initializations
        for n in list(G.nodes()):
            x_s = np.random.standard_normal() #opinion state
            G.nodes[n]["opinion_state"] = x_s

            for j in list(G.neighbors(n)):
                #print('setting weighjt', n_i, n_j)
                w_n = np.random.uniform(0.0, np.nextafter(1,2)) # weights - WHAT ABOUT WEIGHT (5,5)?
                G[j][n]["weight"]= w_n

        parameters = combination
    
        conformity = parameters[0]
        homophily = parameters[1]
        novelty = parameters[2]
        t_h = parameters[3]
        t_a = parameters[4]
        #print(conformity)

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
                        G.nodes[node]["opinion_state"] += conformity * (avg - G.nodes[node]["opinion_state"])  * dt
            
                    #update the node's weight from its neighbourhood
                    for j in neighbours:
                        dif = abs(G.nodes[node]["opinion_state"] - G.nodes[j]["opinion_state"])
                        G[node][j]["weight"] += homophily * (t_h - dif)  * dt
                        if sum2>0:
                            dif = abs(avg - G.nodes[j]["opinion_state"]) 
                            G[node][j]["weight"] += novelty * (dif - t_a) * dt

                        if G[node][j]['weight'] < 0:
                            G[node][j]['weight'] = 0



                G.nodes[node]["opinion_state"] += epsilon
       
        
    
        UG = G.to_undirected()
        for node in list(G.nodes()):
            for neighbour in G.neighbors(node):

                UG.edges[node, neighbour]["weight"] = (
                        (G.edges[node, neighbour]["weight"] + G.edges[neighbour, node]["weight"])/2
                    )
    

        #calculate the average weight of all edges
        s = sum(UG.edges[node, neighbour]["weight"] for node, neighbour in UG.edges)
        avg_weight = s / len(list(UG.edges))
        #print(avg_weight)

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
      
        average_opinion_states = []
        for i in range (len(a)):
            sum_opinion_state = 0
        
            for k in a[i]:
                #calculate the average opinion state of communities
                sum_opinion_state += UG.nodes[k]["opinion_state"]
         
            average_opinion_state = sum_opinion_state / len(a[i])

            average_opinion_states.append(average_opinion_state)
         
        
      
        std_dev = np.std(average_opinion_states)
                
        #calcuate the range of opinion states
        range_community = max(average_opinion_states) - min(average_opinion_states)
       
            

        dict_data = {'h_values':homophily,'a_values':novelty,'c_values': conformity,'th_values':t_h,'ta_values':t_a,'avg_weight': avg_weight, 'std_of_avg_comm_state':std_dev,  'range_of_comm_state': range_community, 'number_of_comm': n_communities,'modularity': modularity}
        file_exists = os.path.isfile('async_1000.csv')

        with open('async_1000.csv', mode='a') as csv_file:
            
                fieldnames = ['h_values','a_values','c_values','th_values','ta_values','avg_weight', 'std_of_avg_comm_state', 'range_of_comm_state', 'number_of_comm','modularity']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader() 
                writer.writerow(dict_data)
                
        #         csv_file.close()

    print(min_edge_weight,max_edge_weight)

# def test():
   
#     lst = []
#     for i in range(100):
#         lst.append(i)

if __name__ == "__main__":
    main_asynch()
    #visualize_graph(G)

    # For Python>=3.5 one can also write:
    # print(timeit.timeit("test()", globals=locals()))

