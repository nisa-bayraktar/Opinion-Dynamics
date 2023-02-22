import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 

# df = pd.read_csv("async.csv")
# grouped = df.groupby(['c_values','h_values','a_values','th_values','ta_values'])

# avg =  grouped.mean()
# avg = avg.reset_index()

# X = avg[['c_values','h_values','a_values','th_values','ta_values']] 
# y_weight = avg['avg_weight']
# y_std = avg['std_of_avg_comm_state']
# y_range = avg['range_of_comm_state']
# y_num = avg['number_of_comm']
# y_mod = avg['modularity']
df = pd.read_csv("async_1.csv")

X = df[['c_values','h_values','a_values','th_values','ta_values']] 
y_weight = df['avg_weight']
y_std = df['std_of_avg_comm_state']
y_range = df['range_of_comm_state']
y_num = df['number_of_comm']
y_mod = df['modularity']

X = sm.add_constant(X) 

# Create a list of the y variables and corresponding est objects
y_list = [y_weight, y_num,y_mod, y_range,y_std]

est_list = [sm.OLS(y, X).fit() for y in y_list ]

coef_list = [est.params for est in est_list]
pval_list = [est.pvalues for est in est_list]
model_names = ['avg_weight', 'number_of_comm','modularity','range_of_comm_state','std_of_avg_comm_state']
x_values =['c_values','h_values','a_values','th_values','ta_values']
# for model in model_names:
#     for i in lenx_values:

#         est2 = smf.ols(formula='{model}'+'~'+ '{x}'+ '*'+'{h_values}' , data=df).fit() 
#         #print(est2.summary())
# Create a table of the coefficients
coef_table = pd.DataFrame(data=coef_list, columns=X.columns, index=model_names)

for i, p in enumerate(pval_list):
    for j, val in enumerate(p):
        if val < 0.00001:
            coef_table.iloc[i,j] = '{:.2f}***'.format(coef_table.iloc[i,j])
        elif val < 0.0001:
            coef_table.iloc[i,j] = '{:.2f}**'.format(coef_table.iloc[i,j])
       
        elif val < 0.001:
            coef_table.iloc[i,j] = '{:.2f}*'.format(coef_table.iloc[i,j])
        else:
            coef_table.iloc[i,j] = '{:.2f}'.format(coef_table.iloc[i,j])
coef_table = coef_table.T

fig, ax = plt.subplots(figsize=(10,5))
ax.axis('off')
ax.axis('tight')
table=ax.table(cellText=coef_table.values, colLabels=coef_table.columns, rowLabels=coef_table.index, loc='center')
fontsize = 14
table.set_fontsize(fontsize)
table.scale(1, 1)

    
# save the plot as a PNG file
plt.savefig('coefficient_table2.png')