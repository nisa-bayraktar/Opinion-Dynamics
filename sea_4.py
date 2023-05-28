import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("assort.csv")
cols = ['aa','hh','cc','ah','ha','ac','ca','ch','hc']

# Calculate standard error for each mean
std_err = df.groupby('method').std() / np.sqrt(10)

# Calculate t-test statistics and p-values for difference from zero
t_stat = df.groupby('method').mean() / std_err
p_values = t.sf(np.abs(t_stat), 9)*2

# Create a bar plot with error bars and asterisks for statistically significant values
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)

sns.barplot(x="method", y="aa", data=df, palette="Reds", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')
sns.barplot(x="method", y="hh", data=df, palette="Blues", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')
sns.barplot(x="method", y="cc", data=df, palette="Greens", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')
sns.barplot(x="method", y="ah", data=df, palette="Purples", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')
sns.barplot(x="method", y="ha", data=df, palette="Oranges", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')
sns.barplot(x="method", y="ac", data=df, palette="Pastel1", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')
sns.barplot(x="method", y="ca", data=df, palette="Pastel2", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')
sns.barplot(x="method", y="ch", data=df, palette="Set1", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')
sns.barplot(x="method", y="hc", data=df, palette="Set2", ax=ax, ci='sd', errwidth=1.5, capsize=0.1, errcolor='black')

for i, method in enumerate(df['method'].unique()):
        x = i - 0.2 + j/10
        y_values = [df[df['method'] == method][col].mean() for col in cols]
        yerrs = [0.1 for _ in range(len(y_values))]
        ax.errorbar(x, y, yerr=yerrs, fmt='none', ecolor='black', capsize=3, capthick=1.5)
        if p_values[cos] < 0.01:
            ax.text(x, y+0.01, '*', ha='center', va='center', fontdict={'size': 16})

# Add labels and a title to the chart
ax.set_ylabel('Correlation Coefficients', fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

