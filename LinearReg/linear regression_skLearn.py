#https://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the diabetes dataset
boston = load_boston()

#print(type(boston))
#print(boston.keys())
#print(boston.DESCR)

data_frame = pd.DataFrame(boston.data)
print(data_frame.head())

# change the column names
data_frame.columns = boston.feature_names
print(data_frame.head())

# Target housing price
data_frame['PRICE'] = boston.target
print(data_frame.head(2))

# Removing the target housing price since it is not necessary
training_data = data_frame.drop('PRICE',axis='columns')
print(training_data.head())

# Create a LR object
linear_mod = LinearRegression()

# Train a MODEL
linear_mod.fit(training_data, data_frame.PRICE)

# Check the correlation of features
additional_info = pd.DataFrame(list(zip(training_data.columns, linear_mod.coef_)), columns = ['features', 'estCoef'])
#print(additional_info)
#plt.scatter(data_frame['RM'], data_frame['PRICE'])
#plt.xlabel('ave numb of rooms per dwelling (RM)')
#plt.ylabel('Housing price')
#plt.title('Relationship bet RM and price')
#plt.show()

#original price
print(data_frame['PRICE'][0:5])
#predicted price
print(linear_mod.predict(training_data)[0:5])

#mean square error
mean_sq_err = np.mean((data_frame['PRICE'] - linear_mod.predict(training_data)) ** 2)
print(mean_sq_err)

## ---------------- ##
# percentage split
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train
