#https://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
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
train_feature, test_feature, train_price, test_price = cross_validation.train_test_split(training_data, data_frame['PRICE'], test_size=0.33, random_state=5)
linear_mod = LinearRegression()
linear_mod.fit(train_feature, train_price)
pred_train = linear_mod.predict(train_feature)
pred_test = linear_mod.predict(test_feature)
print('--------------------')
print('--------------------')
print('\nTrain_feature\n')
print(train_feature.head())
print('\nTest_feature\n')
print(test_feature.head())
print('\nTrain_price\n')
print(train_price.head())
print('\nTest_price\n')
print(test_price.head())
'''
mean_sq_err = np.mean((train_price - linear_mod.predict(train_feature)) ** 2)
print('fit model train_feature and MSE with test_feature', mean_sq_err)
mean_sq_err = np.mean((test_price - linear_mod.predict(test_feature)) ** 2)
print('fit model train_feature and MSE with test_feature, test_price', mean_sq_err)
'''