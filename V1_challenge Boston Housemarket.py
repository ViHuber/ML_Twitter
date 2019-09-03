#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:44:45 2019

@author: viktoriahuber
"""
#Create Dataframe 
from sklearn import datasets
import pandas as pd
import numpy as np

boston = datasets.load_boston()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
df['MEDV'] = boston.target
print(df.head())


import warnings
warnings.simplefilter('ignore')


#Descriptive Statistics
print(df.shape)#nr of rows and colums
print(df.count)
print(df.dtypes)

df.describe(include = 'all') #descriptive statistics

df_grp= df.groupby('MEDV')
df_grp['CRIM'].var()

#Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix= df.corr().round(2)
sns.heatmap(data= correlation_matrix, annot=True)
plt.show()

sns.distplot(df['MEDV'])
plt.show()

plt.hist(df['MEDV'])
plt.title('Histogram for MEDV')
plt.show()

plt.scatter(df['LSTAT'], df['MEDV'])
plt.title('MEDV amount in relation to percentage of lower status population')
plt.xlabel('Low status population')
plt.ylabel('Median House Value')
plt.show()


#Machine Learning 
categorical_names= list(df.select_dtypes(include='category'))
df_ml = pd.get_dummies(df, prefix_sep="__", columns=categorical_names)
df_ml.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_ml.drop(columns = ['MEDV']), df.ix[:, 'MEDV'], 
                                                    test_size=0.3, random_state=23)

print("Number of rows and columns of X_train: ", X_train.shape)
print("Number of rows and columns of X_test: ", X_test.shape)
print("Number of rows and columns of y_train: ", y_train.shape)
print("Number of rows and columns of y_test: ", y_test.shape)


#Lineare Regression 
from sklearn.linear_model import LinearRegression

modelLR = LinearRegression()
modelLR.fit(X_train, y_train)
predictionsLR = modelLR.predict(X_test)
print(modelLR)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

print("MSE for this model is: ", mean_squared_error(y_test, predictionsLR))
print("MAE for this model is: ", mean_absolute_error(y_test, predictionsLR))
print("Explained variance for this model is: ", explained_variance_score(y_test, predictionsLR))

#RandomForest
from sklearn.ensemble import RandomForestRegressor

modelRF = RandomForestRegressor(random_state=23, n_estimators=1000, criterion="mae")
modelRF.fit(X_train, y_train)
predictionsRF = modelRF.predict(X_test)

print("MSE for this model is: ", mean_squared_error(y_test, predictionsRF))
print("MAE for this model is: ", mean_absolute_error(y_test, predictionsRF))
print("Explained variance for this model is: ", explained_variance_score(y_test, predictionsRF))


#Importance of Variables
plt.barh(X_train.columns, modelRF.feature_importances_)
plt.title('Feature Importances of Random Forest')
plt.show()

#Improve Dataset
#Deleting LSTAT, RD,
def function_regression(df, columns_to_drop):
    df1 = df.drop(columns_to_drop, axis=1)
    print(df1)

    #run ML again
    from sklearn.model_selection import train_test_split

    categorical_names= list(df1.select_dtypes(include='category'))
    df_ml = pd.get_dummies(df1, prefix_sep="__", columns=categorical_names)
    df_ml.head()

    X_train, X_test, y_train, y_test = train_test_split(df_ml.drop(columns = ['MEDV']), df1.ix[:, 'MEDV'], 
                                                    test_size=0.3, random_state=23)

    print("Number of rows and columns of X_train: ", X_train.shape)
    print("Number of rows and columns of X_test: ", X_test.shape)
    print("Number of rows and columns of y_train: ", y_train.shape)
    print("Number of rows and columns of y_test: ", y_test.shape)


    modelRF = RandomForestRegressor(random_state=23, n_estimators=1000, criterion="mae")
    modelRF.fit(X_train, y_train)
    predictionsRF = modelRF.predict(X_test)

    print("MSE for this model is: ", mean_squared_error(y_test, predictionsRF))
    print("MAE for this model is: ", mean_absolute_error(y_test, predictionsRF))
    print("Explained variance for this model is: ", explained_variance_score(y_test, predictionsRF))
    
    return df1
    
df_drop3 = function_regression(df, ['CHAS','ZN','RAD'])
#--> Didn't improve the model

df_drop2 = function_regression(df, ['CHAS','ZN'])

#defining the relationship of the X Variables with Y
sns.lmplot(x = 'LSTAT', y = 'MEDV', data = df_drop2)
plt.show()

#Subplots to plot all X on Y together
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(14,4))
flatten = [val for sublist in axes for val in sublist]
for xcol, ax in zip(df_drop2.columns.values.tolist(), flatten):
    df_drop2.plot(kind='scatter', x=xcol, y='MEDV', ax=ax, alpha=0.5, color='b')
    

#Keras Deep Learning Library for Python

#TO BE CONTINUED...