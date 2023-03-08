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

G = pickle.load(open('1000_network_choice/1000_network_final_5', 'rb'))

# sum_w = 0
# sum_a_i= 0
# sum_c_i = 0
# sum_h_i = 0
# sum_h_j = 0
# sum_a_j = 0
# sum_c_j = 0

# multi = 0
# dev_i = 0
# dev_j = 0
  
# for i,j,d in G.edges(data=True):
#         sum_w += d['weight']
#         sum_h_i +=G.nodes[i]['h']
#         sum_a_i +=G.nodes[i]['a']
#         sum_c_i += G.nodes[i]['c']
#         sum_h_j +=G.nodes[j]['h']
#         sum_a_j +=G.nodes[j]['a']
#         sum_c_j += G.nodes[j]['c']


# for i,j,d in G.edges(data=True):
#         diff_i = G.nodes[i]['h'] - (sum_h_i/ sum_w)
#         diff_j =G.nodes[j]['h'] - (sum_h_j/ sum_w)
#         multi += diff_i * diff_j 
#         dev_i += (diff_i)**2 / sum_w
#         dev_j += (diff_j)**2 /sum_w
# sqrt_i = m.sqrt(dev_i)
# sqrt_j = m.sqrt(dev_j)
# pearson_corr = multi / (sum_w * sqrt_i * sqrt_j)
# print(pearson_corr)

set_ai = []
set_hi = []
set_ci = []

set_hj = [] 
set_aj = [] 
set_cj = [] 

set_wij = [] 
sum_w = 0
for i,j,d in G.edges(data=True):
    sum_w += d['weight']
    w_ij = G[i][j]["weight"] 

    a_i = G.nodes[i]["a"]
    h_i = G.nodes[i]["h"]
    c_i = G.nodes[i]["c"]

    h_j = G.nodes[j]["h"]
    a_j = G.nodes[j]["a"]
    c_j = G.nodes[j]["c"]

    X_a = w_ij * a_i
    X_c = w_ij * c_i
    X_h = w_ij * h_i

    Y_h = w_ij * h_j
    Y_c = w_ij * c_j
    Y_a = w_ij * a_j

    set_wij.append(w_ij)

    set_ai.append(X_a)
    set_hi.append(X_h)
    set_ci.append(X_c)

    set_hj.append(Y_h)
    set_aj.append(Y_a)
    set_cj.append(Y_c)

X_bar_a = sum(set_ai) / sum_w
X_bar_c = sum(set_ci) / sum_w
X_bar_h = sum(set_hi) / sum_w

Y_bar_h = sum(set_hj) / sum_w
Y_bar_a = sum(set_aj) / sum_w
Y_bar_c = sum(set_cj) / sum_w


numerator_ah = 0
numerator_ha = 0
numerator_ac = 0
numerator_ca = 0
numerator_ch = 0
numerator_hc = 0

i_sum_a =  0
i_sum_h =  0
i_sum_c =  0

j_sum_h =  0
j_sum_c =  0
j_sum_a =  0

for i in range (len(set_ai)):
    val_i_a = set_ai[i] - X_bar_a
    val_i_c = set_ci[i] - X_bar_c
    val_i_h = set_hi[i] - X_bar_h

    val_j_h = set_hj[i] - Y_bar_h
    val_j_c = set_cj[i] - Y_bar_c
    val_j_a = set_aj[i] - Y_bar_a

    numerator_ah += set_wij[i] * val_i_a * val_j_h
    numerator_ha += set_wij[i] * val_i_h * val_j_a
    numerator_ac += set_wij[i] * val_i_a * val_j_c
    numerator_ca += set_wij[i] * val_i_c * val_j_a
    numerator_ch += set_wij[i] * val_i_c * val_j_h
    numerator_hc += set_wij[i] * val_i_h * val_j_c
 

    i_sum_a += (set_wij[i] * (val_i_a**2)) / sum_w
    i_sum_c += (set_wij[i] * (val_i_c**2)) / sum_w
    i_sum_h += (set_wij[i] * (val_i_h**2)) / sum_w

    j_sum_h += (set_wij[i] * (val_j_h**2)) / sum_w
    j_sum_c += (set_wij[i] * (val_j_c**2)) / sum_w
    j_sum_a += (set_wij[i] * (val_j_a**2)) / sum_w

sigma_i_a = m.sqrt(i_sum_a)
sigma_i_h = m.sqrt(i_sum_h)
sigma_i_c = m.sqrt(i_sum_c)

sigma_j_a = m.sqrt(j_sum_a)
sigma_j_h = m.sqrt(j_sum_h)
sigma_j_c = m.sqrt(j_sum_c)

denominator_ah = sum_w * sigma_i_a * sigma_j_h
denominator_ha = sum_w * sigma_i_h * sigma_j_a
denominator_ac = sum_w * sigma_i_a * sigma_j_c
denominator_ca = sum_w * sigma_i_c * sigma_j_a
denominator_ch = sum_w * sigma_i_c * sigma_j_h
denominator_hc = sum_w * sigma_i_h * sigma_j_c

pearson_corr_ah = numerator_ah / denominator_ah
pearson_corr_ha = numerator_ha / denominator_ha
pearson_corr_ca = numerator_ca / denominator_ca
pearson_corr_ac = numerator_ac / denominator_ac
pearson_corr_ch = numerator_ch / denominator_ch
pearson_corr_hc = numerator_hc / denominator_hc


# pearson_corr_ac = numerator / denominator
# pearson_corr_ch = numerator / denominator
# range = (pearson_corr - (-1))
# sub = ((pearson_corr / range) * 2) - 1
dict_data = {'pearson_corr_ah': pearson_corr_ah,'pearson_corr_ha': pearson_corr_ha,'pearson_corr_ac': pearson_corr_ac,'pearson_corr_ca': pearson_corr_ca,'pearson_corr_ch':pearson_corr_ch,'pearson_corr_hc': pearson_corr_hc}
file_exists = os.path.isfile('assortativity_choice.csv')
with open('assortativity_choice.csv', mode='a') as csv_file:
                
                fieldnames = ['pearson_corr_ah','pearson_corr_ha','pearson_corr_ac','pearson_corr_ca','pearson_corr_ch','pearson_corr_hc']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader() 
                writer.writerow(dict_data)
            






