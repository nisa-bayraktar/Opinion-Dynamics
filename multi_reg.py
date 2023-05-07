import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

# df = pd.read_csv("async.csv")
# grouped = df.groupby(['c','h','a','theta_h','theta_a'])

# avg =  grouped.mean()
# avg = avg.reset_index()

# X = avg[['c','h','a','theta_h','theta_a']] 
# y_weight = avg['average edge weight']
# y_std = avg['std. of average comm. states']
# y_range = avg['range of average comm. states']
# y_num = avg['number of communities']
# y_mod = avg['modularity']
df = pd.read_csv("async_1.csv")

X = df[['c','h','a','theta_h','theta_a']] 
X['c:h'] = X['c'] * X['h']
X['c:a'] = X['c'] * X['a']
X['c:theta_h'] = X['c'] * X['theta_h']
X['c:theta_a'] = X['c'] * X['theta_a']
X['h:a'] = X['h'] * X['a']
X['h:theta_h'] = X['h'] * X['theta_h']
X['h:theta_a'] = X['h'] * X['theta_a']
X['a:theta_h'] = X['a'] * X['theta_h']
X['a:theta_a'] = X['a'] * X['theta_a']
X['theta_h:theta_a'] = X['theta_h'] * X['theta_a']
y_weight = df['average edge weight']
y_std = df['std. of average comm. states']
y_range = df['range of average comm. states']
y_num = df['number of communities']
y_mod = df['modularity']

X = sm.add_constant(X) 

# Create a list of the y variables and corresponding est objects
y_list = [y_weight, y_num,y_mod, y_range,y_std]

est_list = [sm.OLS(y, X).fit() for y in y_list ]

coef_list = [est.params for est in est_list]
pval_list = [est.pvalues for est in est_list]
model_names = ['average edge weight', 'number of communities','modularity','range of average comm. states','std. of average comm. states']
x_values =['c','h','a','theta_h','theta_a']


coef_table = pd.DataFrame(data=coef_list, columns=X.columns, index=model_names)
coef_table = coef_table.T

coef_table_c = coef_table.copy()

for i, p in enumerate(pval_list):
    for j, val in enumerate(p):
        if val < 0.00001:
            coef_table_c.iloc[j,i] = '{:.6f}***'.format(coef_table_c.iloc[j,i])
        elif val < 0.0001:
            coef_table_c.iloc[j,i] = '{:.6f}**'.format(coef_table_c.iloc[j,i])
       
        elif val < 0.001:
            coef_table_c.iloc[j,i] = '{:.6f}*'.format(coef_table_c.iloc[j,i])
        else:
            coef_table_c.iloc[j,i] = '{:.6f}'.format(coef_table_c.iloc[j,i])

max_values = coef_table.iloc[:6].max()
min_values = coef_table.iloc[:6].min()



fig, ax = plt.subplots(figsize=(12,5))
ax.axis('off')
ax.axis('tight')

plt.rcParams['font.style'] = 'italic'


table=ax.table(cellText=coef_table_c.values, colLabels=coef_table_c.columns, cellLoc='center',rowLabels=coef_table_c.index, loc='center',edges='open')
table.set_fontsize(12)

# Set the color of the text in the cells
for j in range(1, coef_table_c.shape[0] +1):
    for i in range(coef_table_c.shape[1]):
        if coef_table.iloc[j - 1, i] == max_values[i]:
            table[(j, i)].get_text().set_color('red')
        elif coef_table.iloc[j - 1, i] == min_values[i]:
            table[(j, i)].get_text().set_color('blue')
# create a new table with interaction values
# for i in range(coef_table_c.shape[1]):
#     table[(0, i)].set_edgecolor('red') 
#     table[(5, i)].set_edgecolor('red') 
#     table[(coef_table_c.shape[0]-1, i)].set_edgecolor('blue')


table.scale(1, 1)
plt.show()
    
