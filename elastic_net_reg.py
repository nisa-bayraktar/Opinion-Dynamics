import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Read data from CSV file
df = pd.read_csv("1000_network_normal_0.03.csv")

# Define predictors and response variables
X = df[['c_values', 'h_values', 'a_values']] 
y_global_ecc = df['global_node_eccentricity']
y_within_ecc = df['within_community_eccentricity']
y_comm_op_ecc = df['community_eccentricity']
y_comm_size = df['community_size']

# Add constant to the predictor variables
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_global_ecc, test_size=0.2, random_state=42)

# Create an instance of the Elastic Net Regression model
alpha = 0.5  # The alpha parameter controls the mix of L1 and L2 regularization
enet = ElasticNet(alpha=alpha, fit_intercept=False)

# Fit the model to the training data
enet.fit(X_train, y_train)

# Make predictions on the test data
y_pred = enet.predict(X_test)

# Calculate the mean squared error on the test data
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print the coefficients and p-values
results = sm.OLS(y_global_ecc, X).fit_regularized(alpha=alpha, L1_wt=0.5)
print(results.params)
print("P-values:\n", results.tvalues.apply(lambda x: 2*(1-stats.t.cdf(abs(x), results.df_resid))) )
