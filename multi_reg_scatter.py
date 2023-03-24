import statsmodels.api as sm 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 



df = pd.read_csv("1000_network_normal_0.3.csv")

X = df[['c_values','h_values','a_values']] 
X['c:h'] = X['c_values'] * X['h_values']
X['c:a'] = X['c_values'] * X['a_values']
X['h:a'] = X['h_values'] * X['a_values']

y_global_ecc = df['global_node_eccentricity']
y_local_ecc = df['local_node_eccentricity']
y_comm_op_ecc = df['community_eccentricity']
y_comm_size = df['community_size']


# Create a list of the y variables and corresponding est objects
y_list = [y_global_ecc,y_local_ecc,y_comm_op_ecc,y_comm_size]

for i in range(len(y_list)):
    y = y_list[i]
    est = sm.OLS(y, X).fit()
    coef = est.params
    pval = est.pvalues
   
    for j in range(len(X.columns)):
        x_col = X.columns[j]
        y_col = y.name
        
      
        weights = X[x_col] * coef[j]
        
        sns.scatterplot(data=df, x=x_col, y=y_col, size=weights, sizes=(50, 200))
        plt.title(f"{y_col} vs {x_col}")
        plt.show()

