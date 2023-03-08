
import pandas as pd 
import numpy as np
from scipy.stats import ttest_1samp

df = pd.read_csv("assortativity_choice.csv")


for i in range(df.shape[1]):
    t_statistic, p_value = ttest_1samp(df.iloc[:, i],0)
    print("T-Statistic:", t_statistic)
    print("P-Value:", '{:f}'.format(p_value))
