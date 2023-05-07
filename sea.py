import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('avg_assortativity.csv')

# Set the 'Method' column as the index of the DataFrame
data = data.set_index('method')

# Get the range of indexes for the columns aa to hc
cols = data.columns[0:]

# Convert the DataFrame into a long format using melt
melted_data = data.reset_index().melt(id_vars=['method'], value_vars=cols)

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)

# Use Seaborn's barplot method to create a grouped bar chart
sns.barplot(x='variable', y='value', hue='method', data=melted_data, ax=ax,palette='Blues')

# Add labels to the chart
ax.set_ylabel('average assortativity values',fontsize=14)
ax.legend(title=None,prop={'size': 14})

ax.tick_params(axis='y', labelsize=14)

for label in ax.xaxis.get_ticklabels():
    label.set_fontstyle('italic')
    label.set_fontsize(14)


plt.show()
