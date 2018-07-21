import math
import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

#Generate data
def gendata():
#    obs_1=pd.read_csv('19Jancluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_2=pd.read_csv('30Jancluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_3=pd.read_csv('06febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_4=pd.read_csv('13febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_5=pd.read_csv('20febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_6=pd.read_csv('28febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_7=pd.read_csv('6marCluster.csv',usecols=[1,4],header=None,skiprows=[0])
    frames=[obs_2,obs_3,obs_4,obs_5,obs_6,obs_7]
    obs=pd.concat(frames,ignore_index=True)
    return obs.values

#Generate GMM model and fit the data
def gengmm(nc=4, n_iter = 2):
    g = mixture.GMM(n_components=nc)  # number of components
    g.init_params = ""  # No initialization
    g.n_iter = n_iter   # iteration of EM method
    return g

def update_ngrdi(g,labels):
#    obs_1=pd.read_csv('19Jancluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_2=pd.read_csv('30Jancluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_3=pd.read_csv('06febcluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_4=pd.read_csv('13febcluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_5=pd.read_csv('20febcluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_6=pd.read_csv('28febcluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_7=pd.read_csv('6marCluster.csv',usecols=[3],header=None,skiprows=[0])
    frames=[obs_2,obs_3,obs_4,obs_5,obs_6,obs_7]
    obs=pd.concat(frames,ignore_index=True)
    new_obs=obs
#    new_obs['cx']=[g.means_[i][0] for i in labels]
#    new_obs['cy']=[g.means_[i][1] for i in labels]
    new_obs['cluster']=[i for i in labels]
    print(new_obs)
    return new_obs

obs = gendata()
g = gengmm(4, 100)
ft=g.fit(obs)

## Clusters of all points
labels=ft.predict(obs)
print(labels)
fig=plt.figure()
probs=ft.predict_proba(obs)

## Plotting points size according to the probability in the cluster
size = 50 * probs.max(1) ** 2

## Showing Clusters of Data
plt.scatter(obs[:,0],obs[:,1],c=labels,s=size,cmap='viridis')
plt.show()

## Cluster Centroid Locations
for i in range(4):
    print("Cluster-",(i)," -> Centroid : ",g.means_[i])

### Saving image and Printing Probabilities of data in a cluster
#fig.savefig('ndvi_spad_cluster_all_wt_size.png')
#np.set_printoptions(threshold=np.inf)
#print(probs.round(5))



#######  Using Ngrdi and Cluster Centroid Locations to train data

data=update_ngrdi(g,labels)
data1=data.values
g=gengmm(4,100)
ft=g.fit(data1)

## Clusters of all points
labels1=ft.predict(data1)
print(labels1)
fig=plt.figure()
probs=ft.predict_proba(data1)

## Plotting points size according to the probability in the cluster
size = 50 * probs.max(1) ** 2

## Showing Clusters of Data
plt.scatter(data1[:,0],data1[:,1],c=labels1,s=size,cmap='viridis')
plt.show()

## Train and Testing by splitting data
train= data.sample(frac=0.95, random_state=2)
test= data.loc[~data.index.isin(train.index)]

## for Converting df column into 2 - D array
#train[1] = train[1].as_matrix().reshape(len(train),1)
print(train.shape)
print(test.shape)

## Using Linear Regression
#model= LinearRegression()

## Using Logistic Regression
model = LogisticRegression()

## For using Random forest algorithm
#model= RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

model.fit(train[3].values.reshape(len(train),1), train['cluster'].values.reshape(len(train),1))

## for predicting
predictions = model.predict(test[3].values.reshape(len(test),1))
#print(mean_squared_error(predictions, test['cluster'].values.reshape(len(test),1)))

## accuracy
#accuracy = model.score(test[3].values.reshape(len(test),1),test['cluster'].values.reshape(len(test),1))
#print(abs(accuracy)*100,'%')

print(test[3].values)
print(test['cluster'].values)
print(predictions)
