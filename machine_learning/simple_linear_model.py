simple linear regresion 

ITS FITS the best fit line


Instantiate Linear Regression model
# Learns relationship: y = (coef * X) + intercept
model = LinearRegression()


# Reshape feature into 2D array as required by scikit-learn
X = data[['feature']]   # Independent variable (predictor)
y = data['target']      # Dependent variable (response)
