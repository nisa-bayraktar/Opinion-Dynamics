import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



df = pd.read_csv("async.csv")

# group the dataframe by the values 
grouped = df.groupby(['c_values','h_values','n_values','th_values','ta_values'])
mask =(df['c_values'] == 0.01) & (df['n_values'] == 0.01)& (df['h_values'] == 0.01)& (df['th_values'] == 0.01)& (df['ta_values'] == 0.01)
#print(df[mask])
# sum = 0
# for c in grouped.groups: 
#     print(c)
#     sum += 1
# print(sum)

print((df.groupby(['c_values','h_values','n_values','th_values','ta_values']).count()).sum())
avg =  grouped.mean()

avg = avg.reset_index()

mask2 =(avg['c_values'] == 0.3) & (avg['n_values'] == 0.3)& (avg['h_values'] == 0.3)& (avg['th_values'] == 0.3)& (avg['ta_values'] == 0.3)
#print(avg[mask2])

X = avg[['n_values']]
#print(X)
y = avg['avg_weight']
#print(y)

# create a LinearRegression object
reg = LinearRegression().fit(X, y)



# make predictions using the regression model
y_pred = reg.predict(X)

#print("Coefficients:", reg.coef_)

plt.scatter(X, y, color='blue',alpha=0.02)
plt.plot(X, y_pred, color='red')
plt.xlabel("novelty values")
plt.ylabel("average weight")
plt.title("Linear Regression")




plt.xticks([0.01, 0.03,0.1,0.3])


plt.show()

