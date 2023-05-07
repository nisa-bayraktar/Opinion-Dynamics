import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress



df = pd.read_csv("new_1000_network_uniform_0.03.csv")

# group the dataframe by the values 
grouped_h = df.groupby(['h','c','a'])
grouped_c = df.groupby(['c','h','a'])
grouped_a = df.groupby(['a','h','c'])




avg_h =  grouped_h.mean()
avg_h = avg_h.reset_index()
avg_c =  grouped_c.mean()
avg_c = avg_c.reset_index()
avg_a =  grouped_a.mean()
avg_a = avg_a.reset_index()




X_c = avg_c['c']
X_h = avg_h['h']
X_a = avg_a['a']



X_values = [X_c, X_h, X_a]


def set_avg(X):
    global avg
    if X is X_c:
        avg = avg_c
    elif X is X_h: 
        avg = avg_h
    elif X is X_a :
        avg = avg_a
   

   

y_values = ['global_node_eccentricity','within_community_eccentricity','community_eccentricity','community_size']



fig, axs = plt.subplots(len(X_values), len(y_values),figsize=(10,12))

# set labels for y axis
for ax, y in zip(axs[:,0], y_values):
    ax.set_ylabel(y, fontsize=10)

# set labels for x axis
for ax, X in zip(axs[-1], X_values):
    ax.set_xlabel(X.name, fontsize=10,style='italic')

for j, X in enumerate(X_values):
    set_avg(X)
    for i, y in enumerate(y_values):
        slope, intercept, r_value, p_value, std_err = linregress(X, avg[y])
        if p_value < 0.00003:
            y_pred = slope * X + intercept
            axs[i,j].scatter(X, avg[y], color='blue', alpha=0.02)
            axs[i,j].plot(X, y_pred, color='red')
            axs[i,j].set_xscale("log")
            axs[i,j].set_xticks([0.01, 0.03, 0.1, 0.3])
            axs[i,j].set_xticklabels([0.01, 0.03, 0.1, 0.3])
            axs[i,j].text(0.95, 0.95, 'p < 0.00001', transform=axs[i,j].transAxes,
                          verticalalignment='top', horizontalalignment='right',
                          fontsize=8, color='black')
        
            
        else:
            axs[i,j].scatter(X, avg[y], color='blue', alpha=0.02)
            axs[i,j].set_xscale("log")
            axs[i,j].set_xticks([0.01, 0.03, 0.1, 0.3])
            axs[i,j].set_xticklabels([0.01, 0.03, 0.1, 0.3])
            # axs[i,j].text(0.95, 0.95, 'p < 0.00001', transform=axs[i,j].transAxes,
            #               verticalalignment='top', horizontalalignment='right',
            #               fontsize=8, color='black')
           
     
plt.tight_layout()
plt.savefig('linear_reg_ex.png')


