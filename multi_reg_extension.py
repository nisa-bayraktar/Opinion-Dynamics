import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf 
import matplotlib


df = pd.read_csv("new_1000_network_uniform_0.03.csv")

X = df[['c_values','h_values','a_values']] 
X['c:h'] = X['c_values'] * X['h_values']
X['c:a'] = X['c_values'] * X['a_values']
X['h:a'] = X['h_values'] * X['a_values']

y_global_ecc = df['global_node_eccentricity']
y_within_ecc = df['within_community_eccentricity']
y_comm_op_ecc = df['community_eccentricity']
y_comm_size = df['community_size']


X = sm.add_constant(X) 


# Create a list of the y variables and corresponding est objects
y_list = [y_global_ecc,y_within_ecc,y_comm_op_ecc,y_comm_size]

est_list = [sm.OLS(y, X).fit() for y in y_list ]

coef_list = [est.params for est in est_list]
pval_list = [est.pvalues for est in est_list]


model_names = ['global_node_eccentricity','within_community_eccentricity', 'community_eccentricity','community_size']
x_values =['c_values','h_values','a_values']


coef_table = pd.DataFrame(data=coef_list, columns=X.columns, index=model_names)
coef_table = coef_table.T
#copy the coef table to put starts, because using the actual table for both colouring the min-max values with stars does not work
coef_table_c = coef_table.copy()


for i, p in enumerate(pval_list):
    print(i,p)
    for j, val in enumerate(p):
        print(j,val)
        if val < 0.0001:
            coef_table_c.iloc[j,i] = '{:.2f}***'.format(coef_table_c.iloc[j,i])
        elif val < 0.001:
            coef_table_c.iloc[j,i] = '{:.2f}**'.format(coef_table_c.iloc[j,i])
       
        elif val < 0.01:
            coef_table_c.iloc[j,i] = '{:.2f}*'.format(coef_table_c.iloc[j,i])
        else:
            coef_table_c.iloc[j,i] = '{:.2f}'.format(coef_table_c.iloc[j,i])

max_values = coef_table.iloc[1:4].max()
min_values = coef_table.iloc[1:4].min()



fig, ax = plt.subplots(figsize=(10,5))
ax.axis('off')
ax.axis('tight')

table=ax.table(cellText=coef_table_c.values, colLabels=coef_table_c.columns, cellLoc='center',rowLabels=coef_table_c.index, loc='center')
fontsize = 12
table.set_fontsize(fontsize)

# Set the color of the text in the cells
for j in range(1, coef_table_c.shape[0] +1):
    for i in range(coef_table_c.shape[1]):
        if coef_table.iloc[j - 1, i] == max_values[i]:
            table[(j, i)].get_text().set_color('red')
        elif coef_table.iloc[j - 1, i] == min_values[i]:
            table[(j, i)].get_text().set_color('blue')


# table.scale(1, 1)
# plt.show()
     
plt.tight_layout()
plt.savefig('multi_reg_uniform.png')
