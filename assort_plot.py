import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from scipy.stats import sem


df = pd.read_csv("NEW_assortativity.csv")
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)

barplot_data = pd.melt(df, id_vars=["method"], value_vars=["aa","hh","cc","ah","ha","ac","ca","ch","hc"])


ax = sns.barplot(x="method", y="value", hue="variable",ax=ax, data=barplot_data,palette="Reds")
std_err = barplot_data.groupby("variable")["value"].apply(sem).tolist()
bar_containers = ax.containers

num_cont = len(bar_containers)
groups = ["aa", "hh", "cc", "ah", "ha", "ac", "ca", "ch", "hc"]
for i in range(num_cont):
    bar_container = bar_containers[i]
    num_bar = len(bar_container)
    x_values = []
    y_values = []
    for i in range(num_bar):
        bar = bar_container[i]
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.errorbar(x, y, yerr=std_err[i], capsize=8, color='black')
        t_value = (y / std_err[i]) - 0 
        if abs(t_value) >= 2.821:  # Check condition for asterisk
            ax.text(x +0.05, y, "*", ha='center', va='bottom', fontsize=14, fontweight = 'bold', color='blue')
        elif abs(t_value) >= 4.297:  # Check condition for asterisk
            ax.text(x, y + 0.02, "**", ha='center', va='bottom', fontsize=14)  # Add asterisk

ax.set_ylabel('assortativity values',fontsize=14)

ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.legend(title=None, prop={'size': 14, 'style': 'italic'})


plt.show()