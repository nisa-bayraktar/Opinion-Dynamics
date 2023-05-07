import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("avg_assortativity.csv")


df = pd.DataFrame({
    'method': ['normal dist.', 'uniform dist.', 'random choice'],
    'aa': [0.24, 0.33, 0.12],
    'ah': [0.27, 0.32, 0.18],
    'hc': [0.30, 0.27, 0.22],
    'hh': [0.34, 0.30, 0.28],
    'cc': [0.36, 0.31, 0.25],
    'ca': [0.38, 0.32, 0.30],
    'ac': [0.40, 0.35, 0.32],
    'ch': [0.43, 0.38, 0.36]
})

# Set the 'Method' column as the index of the DataFrame
df.set_index('method', inplace=True)

# Define the color palette
blue_palette = ['#0343DF', '#E50000', '#030764', '#A52A2A', '#069AF3', '#650021', '#008080' ,'#FF6347', '#7BC8F6']

# Create a bar plot using seaborn
ax = sns.barplot(data=df, palette='Blues')

# Set the y-axis label and title
ax.set_ylabel('average assortativity values')

# Set the x-axis label and ticks
ax.set_xlabel('method')
ax.set_xticklabels(['normal dist.','uniform dist.','random choice'])

# Add the legend
legend = ax.legend()
legend.get_texts()[0].set_text('aa')
legend.get_texts()[1].set_text('ah')
legend.get_texts()[2].set_text('hc')
legend.get_texts()[3].set_text('hh')
legend.get_texts()[4].set_text('cc')
legend.get_texts()[5].set_text('ca')
legend.get_texts()[6].set_text('ac')
legend.get_texts()[7].set_text('ch')

# Make the legend text italic
for text in legend.get_texts():
    text.set_fontstyle('italic')

# Show the plot
plt.show()

