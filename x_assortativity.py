from pyvis.network import Network
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import networkx.algorithms.community as nx_comm
import csv
import os.path
import timeit
import itertools
import pickle as pickle
import math as m

network_files = ['new_uniform_network_0.03/new_1000_network_uniform_1', 'new_uniform_network_0.03/new_1000_network_uniform_2', 'new_uniform_network_0.03/new_1000_network_uniform_3','new_uniform_network_0.03/new_1000_network_uniform_4','new_uniform_network_0.03/new_1000_network_uniform_5','new_uniform_network_0.03/new_1000_network_uniform_6','new_uniform_network_0.03/new_1000_network_uniform_7','new_uniform_network_0.03/new_1000_network_uniform_8','new_uniform_network_0.03/new_1000_network_uniform_9','new_uniform_network_0.03/new_1000_network_uniform_10']
networks = []
for network_file in network_files:
    with open(network_file, 'rb') as f:
        network = pickle.load(f)
        networks.append(network)

# Combine the networks into a single graph
G = nx.compose_all(networks)


set_xi = []
set_ai = []
set_hi = []
set_ci = []

set_xj = []
set_hj = [] 
set_aj = [] 
set_cj = [] 

set_w_xi = []
set_w_ai = []
set_w_hi = []
set_w_ci = []

set_w_xj = []
set_w_hj = [] 
set_w_aj = [] 
set_w_cj = [] 

set_wij = [] 
sum_w = 0

for i,j,d in G.edges(data=True):
    #all the weights in the graph
    sum_w += d['weight']

    #gets the weights in both ways
    w_ij = G[i][j]["weight"] 
    w_ji = G[j][i]["weight"] 
    
    #gets the i node values
    x_i = G.nodes[i]["opinion_state"]
    a_i = G.nodes[i]["a"]
    h_i = G.nodes[i]["h"]
    c_i = G.nodes[i]["c"]
    #gets the j node values
    x_j = G.nodes[j]["opinion_state"]
    x_j = G.nodes[j]["h"]
    h_j = G.nodes[j]["h"]
    a_j = G.nodes[j]["a"]
    c_j = G.nodes[j]["c"]
    #calculates the i values * i weights
    wX_x = w_ij * x_i
    wX_a = w_ij * a_i # wX_a = w_ij * a_i
    wX_c = w_ij * c_i
    wX_h = w_ij * h_i
    #calculates the j values * j weights
    wY_x = w_ji * x_j
    wY_h = w_ji * h_j
    wY_c = w_ji * c_j
    wY_a = w_ji * a_j
    #list of w_ij 
    set_wij.append(w_ij)
    #list of i values * i weights and list of i values itselves
    set_w_xi.append(wX_x)
    set_xi.append(x_i)

    set_w_ai.append(wX_a)
    set_ai.append(a_i)
 
    set_w_hi.append(wX_h)
    set_hi.append(h_i)

    set_w_ci.append(wX_c)
    set_ci.append(c_i)
    #list of j values * j weights and list of j values itselves
    set_w_xj.append(wY_x)
    set_xj.append(x_j)

    set_w_hj.append(wY_h)
    set_hj.append(h_j)

    set_w_aj.append(wY_a)
    set_aj.append(a_j)

    set_w_cj.append(wY_c)
    set_cj.append(c_j)
#calculate the source X_bar (weighted i values/ sum of all the weights(W))
X_bar_x = sum(set_w_xi) / sum_w
X_bar_a = sum(set_w_ai) / sum_w
X_bar_c = sum(set_w_ci) / sum_w
X_bar_h = sum(set_w_hi) / sum_w
#calculate the target Y_bar (weighted j values/ sum of all the weights(W))
Y_bar_x = sum(set_w_xj) / sum_w
Y_bar_h = sum(set_w_hj) / sum_w
Y_bar_a = sum(set_w_aj) / sum_w
Y_bar_c = sum(set_w_cj) / sum_w

numerator_xx = 0
numerator_aa = 0
numerator_hh = 0
numerator_cc = 0
numerator_xa = 0
numerator_ax = 0
numerator_xc = 0
numerator_cx = 0
numerator_xh = 0
numerator_hx = 0
numerator_ah = 0
numerator_ha = 0
numerator_ac = 0
numerator_ca = 0
numerator_ch = 0
numerator_hc = 0

i_sum_x =  0
i_sum_a =  0
i_sum_h =  0
i_sum_c =  0

j_sum_x =  0
j_sum_h =  0
j_sum_c =  0
j_sum_a =  0

for i in range (len(set_ai)):
    #calculates (X_i - X_bar)
    val_i_x = set_xi[i] - X_bar_x
    val_i_a = set_ai[i] - X_bar_a # instead of set_ai[i] we want to use a[i]
    val_i_c = set_ci[i] - X_bar_c
    val_i_h = set_hi[i] - X_bar_h
  
    #calculates (Y_j - Y_bar)
    val_j_x = set_xj[i] - Y_bar_x
    val_j_h = set_hj[i] - Y_bar_h
    val_j_c = set_cj[i] - Y_bar_c
    val_j_a = set_aj[i] - Y_bar_a
    #calculates w_ij * (X_i - X_bar)(Y_j - Y_bar)
    numerator_xx += set_wij[i] * val_i_x * val_j_x
    numerator_aa += set_wij[i] * val_i_a * val_j_a
    numerator_hh += set_wij[i] * val_i_h * val_j_h
    numerator_cc += set_wij[i] * val_i_c * val_j_c
    numerator_ax += set_wij[i] * val_i_a * val_j_x
    numerator_xa += set_wij[i] * val_i_x * val_j_a
    numerator_xh += set_wij[i] * val_i_x * val_j_h
    numerator_hx += set_wij[i] * val_i_h * val_j_x
    numerator_xc += set_wij[i] * val_i_x * val_j_c
    numerator_cx += set_wij[i] * val_i_c * val_j_x
    numerator_ah += set_wij[i] * val_i_a * val_j_h
    numerator_ha += set_wij[i] * val_i_h * val_j_a
    numerator_ac += set_wij[i] * val_i_a * val_j_c
    numerator_ca += set_wij[i] * val_i_c * val_j_a
    numerator_ch += set_wij[i] * val_i_c * val_j_h
    numerator_hc += set_wij[i] * val_i_h * val_j_c
 
#calculates sigma_i and sigma_j
    i_sum_x += (set_wij[i] * (val_i_x**2)) / sum_w
    i_sum_a += (set_wij[i] * (val_i_a**2)) / sum_w
    i_sum_c += (set_wij[i] * (val_i_c**2)) / sum_w
    i_sum_h += (set_wij[i] * (val_i_h**2)) / sum_w

    j_sum_x += (set_wij[i] * (val_j_x**2)) / sum_w
    j_sum_h += (set_wij[i] * (val_j_h**2)) / sum_w
    j_sum_c += (set_wij[i] * (val_j_c**2)) / sum_w
    j_sum_a += (set_wij[i] * (val_j_a**2)) / sum_w

sigma_i_x = m.sqrt(i_sum_x)
sigma_i_a = m.sqrt(i_sum_a)
sigma_i_h = m.sqrt(i_sum_h)
sigma_i_c = m.sqrt(i_sum_c)

sigma_j_x = m.sqrt(j_sum_x)
sigma_j_a = m.sqrt(j_sum_a)
sigma_j_h = m.sqrt(j_sum_h)
sigma_j_c = m.sqrt(j_sum_c)
#calculates denominator sum of all the weights(W) * sigma_i * sigma_j
denominator_xx = sum_w * sigma_i_x * sigma_j_x
denominator_aa = sum_w * sigma_i_a * sigma_j_a
denominator_cc = sum_w * sigma_i_c * sigma_j_c
denominator_hh = sum_w * sigma_i_h * sigma_j_h
denominator_xa = sum_w * sigma_i_x * sigma_j_a
denominator_ax = sum_w * sigma_i_a * sigma_j_x
denominator_xh = sum_w * sigma_i_x * sigma_j_h
denominator_hx = sum_w * sigma_i_h * sigma_j_x
denominator_xc = sum_w * sigma_i_x * sigma_j_c
denominator_cx = sum_w * sigma_i_c * sigma_j_x
denominator_ah = sum_w * sigma_i_a * sigma_j_h
denominator_ha = sum_w * sigma_i_h * sigma_j_a
denominator_ac = sum_w * sigma_i_a * sigma_j_c
denominator_ca = sum_w * sigma_i_c * sigma_j_a
denominator_ch = sum_w * sigma_i_c * sigma_j_h
denominator_hc = sum_w * sigma_i_h * sigma_j_c
#calculates the pearson correlation
pearson_corr_xx = numerator_xx / denominator_xx
pearson_corr_aa = numerator_aa / denominator_aa
pearson_corr_cc = numerator_cc / denominator_cc
pearson_corr_hh = numerator_hh / denominator_hh
pearson_corr_xa = numerator_xa / denominator_xa
pearson_corr_ax = numerator_ax / denominator_ax
pearson_corr_xh = numerator_xh / denominator_xh
pearson_corr_hx = numerator_hx / denominator_hx
pearson_corr_xc = numerator_xc / denominator_xc
pearson_corr_cx = numerator_cx / denominator_cx
pearson_corr_ah = numerator_ah / denominator_ah
pearson_corr_ha = numerator_ha / denominator_ha
pearson_corr_ca = numerator_ca / denominator_ca
pearson_corr_ac = numerator_ac / denominator_ac
pearson_corr_ch = numerator_ch / denominator_ch
pearson_corr_hc = numerator_hc / denominator_hc

dict_data = {'pearson_corr_xx': pearson_corr_xx,'pearson_corr_aa': pearson_corr_aa,'pearson_corr_hh': pearson_corr_hh,'pearson_corr_cc': pearson_corr_cc,'pearson_corr_xa': pearson_corr_xa,'pearson_corr_ax': pearson_corr_ax,'pearson_corr_xh': pearson_corr_xh,'pearson_corr_hx': pearson_corr_hx,'pearson_corr_xc': pearson_corr_xc,'pearson_corr_cx': pearson_corr_cx,'pearson_corr_ah': pearson_corr_ah,'pearson_corr_ha': pearson_corr_ha,'pearson_corr_ac': pearson_corr_ac,'pearson_corr_ca': pearson_corr_ca,'pearson_corr_ch':pearson_corr_ch,'pearson_corr_hc': pearson_corr_hc}
file_exists = os.path.isfile('combined_new_assortativity_uniform.csv')
with open('combined_new_assortativity_uniform.csv', mode='a') as csv_file:
                
                fieldnames = ['pearson_corr_xx','pearson_corr_aa','pearson_corr_hh','pearson_corr_cc','pearson_corr_xa','pearson_corr_ax','pearson_corr_xh','pearson_corr_hx','pearson_corr_xc','pearson_corr_cx','pearson_corr_ah','pearson_corr_ha','pearson_corr_ac','pearson_corr_ca','pearson_corr_ch','pearson_corr_hc']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader() 
                writer.writerow(dict_data)
            







