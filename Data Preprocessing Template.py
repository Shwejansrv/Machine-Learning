#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values

#missing Data
from sklearn.impute import SimpleImputer
simputer = SimpleImputer(missing_values= np.nan,strategy="mean", verbose=0)
simputer.fit(X[:, 1:3])
X[:, 1:3] = simputer.transform(X[:, 1:3])

#categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X),dtype=np.float)
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

#traintest
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Featurescaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
