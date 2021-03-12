# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:28:48 2020

@author: Dragon_Slayer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init = 'k-means++',max_iter=300,n_init = 10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('No.of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s = 100,c= "red",label = 'Cluster 1')

plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s = 100,c= "blue",label = 'Cluster 2')

plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s = 100,c= "green",label = 'Cluster 3')

plt.scatter(x[y_kmeans == 3,0], x[y_kmeans == 3,1], s = 100,c= "cyan",label = 'Cluster 4')

plt.scatter(x[y_kmeans == 4,0], x[y_kmeans == 4,1], s = 100,c= "magenta",label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300,c= "yellow",label = 'Centroids')
plt.title("Cluster of clients")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()