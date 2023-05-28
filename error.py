import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("NEW_assortativity.csv")

# Calculate the mean and standard deviation for each method/variable combination
grouped = df.groupby(["method"]).agg(["mean", "std"])
grouped.reset_index(inplace=True)
grouped.columns = ["method", "mean", "std"]

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)

# Create the bar chart using Seaborn
sns.barplot(x="method", y="mean", hue="variable", data=grouped, palette="Reds", ax=ax)

# Add error bars to the chart using Matplotlib
for i, method in enumerate(grouped["method"].unique()):
    for j, variable in enumerate(grouped["variable"].unique()):
        mean = grouped.loc[(grouped["method"] == method) & (grouped["variable"] == variable), "mean"].values[0]
        std = grouped.loc[(grouped["method"] == method) & (grouped["variable"] == variable), "std"].values[0]
        ax.errorbar(i+(j-0.5)*0.2, mean, yerr=std, fmt='none', capsize=5, ecolor='black')

# Add labels and a title to the chart
ax.set_ylabel('assortativity values',fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.legend(title=None, prop={'size': 14, 'style': 'italic'})

plt.show()
