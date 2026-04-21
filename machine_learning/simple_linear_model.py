# simple linear regresion 

# ITS FITS the best fit lin



# Learns relationship: y = (coef * X) + intercept
model = LinearRegression()


# Reshape feature into 2D array as required by scikit-learn
X = data[['feature']]   # Independent variable (predictor)
y = data['target']     


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
#plt.scatter(X, y, label="Observed Data")

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



print("\n🔹 Model Parameters (Sklearn)")
print("Intercept:", sk_model.intercept_)
print("Coefficients:", sk_model.coef_)





"""
Simple Linear Regression from Scratch
Author: Deepesh Singh
Description:
    Implements simple linear regression using the closed-form solution
    (Normal Equation) without external ML libraries.
"""

import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        self.m = None  # slope
        self.b = None  # intercept

    def fit(self, X, y):
        """
        Train the model using input features and target values.

        Parameters:
        X : array-like, shape (n_samples,)
        y : array-like, shape (n_samples,)
        """

        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Compute means
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Compute slope (m)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.m = numerator / denominator

        # Compute intercept (b)
        self.b = y_mean - self.m * x_mean

    def predict(self, X):
        """
        Predict target values for given input.

        Parameters:
        X : array-like

        Returns:
        Predictions : array-like
        """
        X = np.array(X)
        return self.m * X + self.b

    def score(self, X, y):
        """
        Calculate R² score.

        R² = 1 - (SS_res / SS_tot)
        """
        y_pred = self.predict(X)
        y = np.array(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)



if __name__ == "__main__":
    # Sample dataset
    X = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    # Initialize model
    model = SimpleLinearRegression()

    # Train model
    model.fit(X, y)

    # Predictions
    predictions = model.predict(X)

    print("Slope (m):", model.m)
    print("Intercept (b):", model.b)
    print("Predictions:", predictions)
    print("R2 Score:", model.score(X,y))
# finised 
