# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:31:25 2020

@author: Dragon_Slayer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title("Dendrogram")
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc == 0,0], x[y_hc == 0,1], s = 100,c= "red",label = 'Cluster 1')

plt.scatter(x[y_hc == 1,0], x[y_hc == 1,1], s = 100,c= "blue",label = 'Cluster 2')

plt.scatter(x[y_hc == 2,0], x[y_hc == 2,1], s = 100,c= "green",label = 'Cluster 3')

plt.scatter(x[y_hc == 3,0], x[y_hc == 3,1], s = 100,c= "cyan",label = 'Cluster 4')

plt.scatter(x[y_hc == 4,0], x[y_hc == 4,1], s = 100,c= "magenta",label = 'Cluster 5')
plt.title("Cluster of clients")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()