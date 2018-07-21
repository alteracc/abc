import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

#Generate data
def gendata():
    obs = pd.read_csv('C:/Users/SUMITKR/Documents/python/gmm/20febcluster.csv',usecols=[1,4],header=None,skiprows=[0])

## for finding correlation among each column
#    print(obs.corr()[4])
    return obs

obs1 = gendata()
obs= obs1.values

##  for visualising HISTOGRAM 
#plt.hist(obs[:,0])
#plt.show()
#plt.hist(obs[:,0])
#plt.show()

## Applying K- Means Clustering
kmeans_model= KMeans(n_clusters=4, random_state=1)
kmeans_model.fit(obs)
labels= kmeans_model.labels_

## Using PCA for results of 3-D or more dimensions into 2-D
#pca_2= PCA(2)
#plot_columns=pca_2.fit_transform(obs)
#plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
#plt.show()

## Train and Testing by splitting data
train= obs1.sample(frac=0.8, random_state=1)
test= obs1.loc[~obs1.index.isin(train.index)]

## for Converting df column into 2 - D array
#train[1] = train[1].as_matrix().reshape(len(train),1)
print(train.shape)
print(test.shape)

## Using Linear Regression
model= LinearRegression()
model.fit(train[1].as_matrix().reshape(len(train),1), train[4].as_matrix().reshape(len(train),1))

## for predicting
predictions = model.predict(test[1].as_matrix().reshape(len(test),1))
mean_squared_error(predictions, test[4].as_matrix().reshape(len(test),1))
print(test[1].values)
print(test[4].values)
print(predictions)
