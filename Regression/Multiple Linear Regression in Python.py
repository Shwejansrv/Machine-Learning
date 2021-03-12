# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:16:50 2020

@author: Dragon_Slayer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x),dtype = np.float)

x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm
x = np.append(arr=np.ones((50,1)).astype(int),values = x,axis = 1)

x_opt = x[:,[0,1,2,3,4,5]]
x_opt = np.array(x_opt,dtype='float')
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:,[0,1,3,4,5]]
x_opt = np.array(x_opt,dtype='float')
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())


x_opt = x[:,[0,3,4,5]]
x_opt = np.array(x_opt,dtype='float')
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())


x_opt = x[:,[0,3,5]]
x_opt = np.array(x_opt,dtype='float')
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())


x_opt = x[:,[0,3]]
x_opt = np.array(x_opt,dtype='float')
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())






