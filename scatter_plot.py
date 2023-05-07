import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df  = pd.read_csv("avg_assortativity.csv")
# Set the 'Method' column as the index of the DataFrame


# Set the 'Method' column as the index of the DataFrame
df.set_index('method', inplace=True)


# Create a figure and axis object
fig, ax = plt.subplots()

# Set the width of each bar
#width = 0.25
width = 0.1

# # Set the positions of the bars on the x-axis
# pos_aa = np.arange(len(df.index))
# pos_hh = [x + width for x in pos_aa]
# pos_cc = [x + width for x in pos_hh]
# pos_ah = [x + width for x in pos_cc]
# pos_ha = [x + width for x in pos_ah]
# pos_ac = [x + width for x in pos_ha]
# pos_ca = [x + width for x in pos_ac]
# pos_ch = [x + width for x in pos_ca]
# pos_hc = [x + width for x in pos_ch]
# width = 0.1

# Set the positions of the bars on the x-axis
num_cols = len(df.columns)
midpoint = num_cols / 2
pos = np.arange(len(df.index)) - midpoint * width

blue_palette = ['#0343DF', '#E50000','#030764','#A52A2A', '#069AF3','#650021','#008080' ,'#FF6347','#7BC8F6']
# Plot the bars for each column
for i, col in enumerate(df.columns):
    vals = df[col].values
    ax.bar(pos + i * width, vals, width, label=col,color=blue_palette[i % len(blue_palette)])
  

# # Plot the bars for each column
# for i, col in enumerate(df.columns):
#     pos = globals()[f"pos_{col.lower()}"]
#     vals = df[col].values
#     ax.bar(pos, vals, width, label=col)
  
# Set the x-axis labels and ticks

ax.set_xticks([0,1,2])
ax.set_xticklabels(['normal dist.','uniform dist.','random choice'], fontstyle='italic')

# Set the y-axis label and title
ax.set_ylabel('average assortativity values')




# Add the legend
ax.legend()


# Show the chart
plt.show()

  
# # Plot the data using bar() method
# plt.bar(X, Y, color='g')
# plt.title("Students over 11 Years")
# plt.xlabel("Years")
# plt.ylabel("Number of Students")
  
# # Show the plot
# plt.show()


# X = df[['aa','hh','cc','ah','ha','ac','ca','ch','hc']] 
# y_data = len(df.index)
# plt.scatter(x=X,y=y_data)
# #df.plot()  # plots all columns against index
# plt.show()
# #df.plot(kind='scatter',x='x',y='y') # scatter plot
# #df.plot(kind='density')  # estimate density function
# # df.plot(kind='hist')  # histogram
