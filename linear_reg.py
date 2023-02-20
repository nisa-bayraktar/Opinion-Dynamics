import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress



df = pd.read_csv("output.csv")

# group the dataframe by the values 
grouped = df.groupby(['c_values','h_values','a_values','th_values','ta_values'])
#mask =(df['c_values'] == 0.01) & (df['n_values'] == 0.01)& (df['h_values'] == 0.01)& (df['th_values'] == 0.01)& (df['ta_values'] == 0.01)
#print(df[mask])
# sum = 0
# for c in grouped.groups: 
#     print(c)
#     sum += 1
# print(sum)

#print((df.groupby(['c_values','h_values','a_values','th_values','ta_values']).count()).sum())
avg =  grouped.mean()

avg = avg.reset_index()

#mask2 =(avg['c_values'] == 0.3) & (avg['n_values'] == 0.3)& (avg['h_values'] == 0.3)& (avg['th_values'] == 0.3)& (avg['ta_values'] == 0.3)
#print(avg[mask2])


X = avg['th_values']

y = avg['avg_weight']


slope, intercept, r_value, p_value, std_err = linregress(X, y)

# Check if the p-value is statistically significant (i.e., less than 0.05)
if p_value < 0.00001:

    # Calculate predicted y-values based on the regression parameters
    y_pred = slope * X + intercept

    # Plot the data points and the regression line
    plt.scatter(X, y, color='blue',alpha=0.02)
    plt.plot(X, y_pred, color='red')
    plt.xlabel("th")
    plt.ylabel("average weight")

    # Add a legend and title
    plt.title('p<0.00001')

    plt.xscale("log")
    x_ticks = [0.01,0.03,0.1,0.3]
    plt.xticks(x_ticks,x_ticks)

    # Show the plot
    plt.show()

else:
    plt.scatter(X, y, color='blue',alpha=0.02)
    plt.xlabel("novelty values")
    plt.ylabel("average weight")
    plt.xscale("log")
    x_ticks = [0.01,0.03,0.1,0.3]
    plt.xticks(x_ticks,x_ticks)
    plt.show()




















# reg = LinearRegression().fit(X, y)



# y_pred = reg.predict(X)





# plt.scatter(X, y, color='blue',alpha=0.02)
# plt.plot(X, y_pred, color='red')

# plt.xlabel("novelty values")
# plt.ylabel("average weight")
# plt.title("Linear Regression")


# plt.xscale("log")
# x_ticks = [0.01,0.03,0.1,0.3]
# plt.xticks(x_ticks,x_ticks)





#plt.show()

