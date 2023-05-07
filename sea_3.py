import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("avg_assortativity.csv")
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)

# Use Seaborn to create the bar chart
ax = sns.barplot(x="method", y="value", hue="variable",ax=ax, data=pd.melt(df, id_vars=["method"]),palette="Reds",)

# Add labels and a title to the chart
ax.set_ylabel('average assortativity values',fontsize=14)

ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.legend(title=None, prop={'size': 14, 'style': 'italic'})


plt.show()