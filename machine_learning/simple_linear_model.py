# simple linear regresion 

# ITS FITS the best fit line


# Instantiate Linear Regression model
# Learns relationship: y = (coef * X) + intercept
model = LinearRegression()


# Reshape feature into 2D array as required by scikit-learn
X = data[['feature']]   # Independent variable (predictor)
y = data['target']      # Dependent variable (response)

# Instantiate Linear Regression model
# Learns relationship: y = (coef * X) + intercept
model = LinearRegression()

# -------------------------------
#  Model Training
# -------------------------------
# Fit model parameters using least squares optimization
model.fit(X, y)

# Generate predictions on the input feature space
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


# Replace with your dataset
# Example: data.csv with columns: ['X1', 'X2', 'X3', 'y']
df = pd.read_csv("data.csv")

# Features (Independent Variables)
X = df.drop(columns=["y"]).values

# Target (Dependent Variable)
y = df["y"].values.reshape(-1, 1)

# ==============================
# ✂️ Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class MultipleLinearRegression:
    """
    Custom implementation using Normal Equation:
    θ = (XᵀX)^(-1) Xᵀy
    """

    def fit(self, X, y):
        # Add bias term (intercept)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute theta using Normal Equation
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b @ self.theta

# Train custom model
custom_model = MultipleLinearRegression()
custom_model.fit(X_train, y_train)

# Predictions
y_pred_custom = custom_model.predict(X_test)

# Evaluation
print("🔹 Custom Model Performance")
print("R2 Score:", r2_score(y_test, y_pred_custom))
print("MSE:", mean_squared_error(y_test, y_pred_custom))

print("\n🔹 Model Parameters (Sklearn)")
print("Intercept:", sk_model.intercept_)
print("Coefficients:", sk_model.coef_)

"""
Interpretation:
- Each coefficient represents the change in target variable (y)
  for a one-unit change in the corresponding feature,
  keeping other features constant.
"""

print("\n🔹 Model Parameters (Sklearn)")
print("Intercept:", sk_model.intercept_)
print("Coefficients:", sk_model.coef_)
