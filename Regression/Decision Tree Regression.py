# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:32:53 2020

@author: Dragon_Slayer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2:3].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

y_pred = regressor.predict([[6.5]])

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y)
plt.plot(x_grid,regressor.predict(x_grid))
plt.title("T vs B(DTR)")
plt.show()


