import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress



df = pd.read_csv("async.csv")

# group the dataframe by the values 
grouped_h = df.groupby(['h_values','c_values','a_values','th_values','ta_values'])
grouped_c = df.groupby(['c_values','h_values','a_values','th_values','ta_values'])
grouped_a = df.groupby(['a_values','h_values','c_values','th_values','ta_values'])
grouped_th = df.groupby(['th_values','h_values','c_values','a_values','ta_values'])
grouped_ta = df.groupby(['ta_values','h_values','c_values','a_values','th_values'])



avg_h =  grouped_h.mean()
avg_h = avg_h.reset_index()
avg_c =  grouped_c.mean()
avg_c = avg_c.reset_index()
avg_a =  grouped_a.mean()
avg_a = avg_a.reset_index()
avg_th =  grouped_th.mean()
avg_th = avg_th.reset_index()
avg_ta =  grouped_ta.mean()
avg_ta = avg_ta.reset_index()



X_c = avg_c['c_values']
X_h = avg_h['h_values']
X_a = avg_a['a_values']
X_th = avg_th['th_values']
X_ta = avg_ta['ta_values']


X_values = [X_c, X_h, X_a, X_th, X_ta]


def set_avg(X):
    global avg
    if X is X_c:
        avg = avg_c
    elif X is X_h: 
        avg = avg_h
    elif X is X_a :
        avg = avg_a
    elif X is X_th :
        avg = avg_th
    else:
        avg = avg_ta

   

y_values = ['avg_weight', 'number_of_comm','modularity','range_of_comm_state','std_of_avg_comm_state']


fig, axs = plt.subplots(len(X_values), len(y_values),figsize=(10,12))

# set labels for y axis
for ax, y in zip(axs[:,0], y_values):
    ax.set_ylabel(y, fontsize=10)

# set labels for x axis
for ax, X in zip(axs[-1], X_values):
    ax.set_xlabel(X.name, fontsize=10)

for j, X in enumerate(X_values):
    set_avg(X)
    for i, y in enumerate(y_values):
        slope, intercept, r_value, p_value, std_err = linregress(X, avg[y])
        if p_value < 0.00001:
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
            axs[i,j].text(0.95, 0.95, 'p < 0.00001', transform=axs[i,j].transAxes,
                          verticalalignment='top', horizontalalignment='right',
                          fontsize=8, color='black')
           
     
plt.tight_layout()
plt.savefig('linear_reg.png')


