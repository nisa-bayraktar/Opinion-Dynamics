import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the CSV file into a pandas DataFrame
data = pd.read_csv('avg_assortativity.csv')

# Set the 'Method' column as the index of the DataFrame
data = data.set_index('method')

# Get the range of indexes for the columns aa to hc
cols = data.columns[1:]
n_cols = len(cols)
indexes = list(range(n_cols))

# Create a figure and axis object
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)

# Set the width of each bar
width = 0.25


# Set the offset for each method
offset = np.linspace(-width/2, width/2, len(data.index))

# Iterate over the rows of the DataFrame and plot a bar chart for each method
for i, (method, row) in enumerate(data.iterrows()):
    x_values = indexes + offset[i]
    y_values = row[cols]

    ax.bar(x_values, y_values, width, label=method)


# Add labels to the chart
ax.set_ylabel('average assortativity values')

ax.set_xticks(indexes)

ax.set_xticklabels(cols)

ax.legend()

# Show the chart
plt.show()


# # Iterate over the rows of the DataFrame and plot a bar chart for each method
# for method, row in data.iterrows():
#     x_values = indexes
#     y_values = row[cols]
#     ax.bar(x_values, y_values, width, label=method)