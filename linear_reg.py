import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



df = pd.read_csv("output.csv")

average_runs = df.groupby(np.arange(len(df.iloc[:, 1:]))//5).mean()
print(average_runs)