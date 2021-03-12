# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:39:27 2020

@author: Dragon_Slayer
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

#Training,testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = X_train.reshape(-1,1)
regressor.fit(X_train,y_train)

X_test = X_test.reshape(-1,1)
y_pred = regressor.predict(X_test)

#visualizing results
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train))
plt.title("SALARY VS EXPERIENCE test" )
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()