import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

#Generate data
def gendata():
    obs_1 = pd.read_csv('20febcluster.csv',usecols=[1,3,4],header=None,skiprows=[0])
    obs_2=pd.read_csv('28febcluster.csv',usecols=[1,3,4],header=None,skiprows=[0])
    obs_3=pd.read_csv('6marCluster.csv',usecols=[1,3,4],header=None,skiprows=[0])
    frames=[obs_1,obs_2,obs_3]
    obs=pd.concat(frames,ignore_index=True)
#    obs.append(obs_2,ignore_index=True)
#    print("Dataframes: \n")
#    print(obs)
## for finding correlation among each column
    print("Correlation : ",obs.corr()[3])
    return obs

obs1 = gendata()
obs= obs1.values
##  for visualising HISTOGRAM 
#plt.hist(obs[:,0])
#plt.show()
#plt.hist(obs[:,0])
#plt.show()

## Applying K- Means Clustering
#kmeans_model= KMeans(n_clusters=4, random_state=1)
#kmeans_model.fit(obs)
#labels= kmeans_model.labels_

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

## For using Random forest algorithm
#model= RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

model.fit(train[3].values.reshape(len(train),1), train[[1,4]].values.reshape(len(train),2))

## for predicting
predictions = model.predict(test[3].values.reshape(len(test),1))
print(mean_squared_error(predictions, test[[1,4]].values.reshape(len(test),2)))
print(test[3].values)
print(test[[1,4]].values)
print(predictions)
