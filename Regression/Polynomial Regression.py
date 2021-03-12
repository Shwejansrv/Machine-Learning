# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:00:20 2020

@author: Dragon_Slayer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y)
plt.plot(x,lin_reg.predict(x))
plt.title("T vs B(LR)")
plt.show()

plt.scatter(x,y)
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)))
plt.title("T vs B(PR)")
plt.show()

print(lin_reg.predict(np.array([[6.5]])))