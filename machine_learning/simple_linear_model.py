simple linear regresion 

ITS FITS the best fit line


Instantiate Linear Regression model
# Learns relationship: y = (coef * X) + intercept
model = LinearRegression()


# Reshape feature into 2D array as required by scikit-learn
X = data[['feature']]   # Independent variable (predictor)
y = data['target']      # Dependent variable (response)

Instantiate Linear Regression model
# Learns relationship: y = (coef * X) + intercept
model = LinearRegression()

# -------------------------------
#  Model Training
# -------------------------------
# Fit model parameters using least squares optimization
model.fit(X, y)

Generate predictions on the input feature space
y_pred = model.predict(X)

# -------------------------------
#  Visualization
# -------------------------------
# Plot original data points
plt.scatter(X, y, label="Observed Data")

# Plot learned regression line
plt.plot(X, y_pred, label="Model Fit")

# Add metadata for interpretability
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Simple Linear Regression")

plt.legend()
plt.show()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

