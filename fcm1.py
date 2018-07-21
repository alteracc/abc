from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
import pydotplus
import collections

colors = ['c', 'orange', 'ForestGreen', 'b', 'm', 'r', 'y', 'k', 'Brown', 'g']

def getdata():
    obs_1=pd.read_csv('12Jancluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_2=pd.read_csv('19Jancluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_3=pd.read_csv('30Jancluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_4=pd.read_csv('06febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_5=pd.read_csv('13febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_6=pd.read_csv('20febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_7=pd.read_csv('28febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_8=pd.read_csv('6marCluster.csv',usecols=[1,4],header=None,skiprows=[0])
    frames=[obs_1,obs_2,obs_3,obs_4,obs_5,obs_6,obs_7,obs_8]
    obs=pd.concat(frames,ignore_index=True)
    obs=obs.dropna()
    return obs.values

def updatedata(cluster_membership):
    obs_1=pd.read_csv('12Jancluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_2=pd.read_csv('19Jancluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_3=pd.read_csv('30Jancluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_4=pd.read_csv('06febcluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_5=pd.read_csv('13febcluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_6=pd.read_csv('20febcluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_7=pd.read_csv('28febcluster.csv',usecols=[3],header=None,skiprows=[0])
    obs_8=pd.read_csv('6marCluster.csv',usecols=[3],header=None,skiprows=[0])
    frames=[obs_1,obs_2,obs_3,obs_4,obs_5,obs_6,obs_7,obs_8]
    obs=pd.concat(frames,ignore_index=True)
    obs=obs.dropna()
    obs['cluster']=[i for i in cluster_membership]
    print(obs)
    return obs

################################################################################
################## -------------- CLUSTERING ----------- #######################
################################################################################

data=getdata()
# Visualize all the data without clusters
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(data[:,0], data[:,1], '.',
             color=colors[label])
ax0.set_title('All NDVI and SPAD data:')

plt.show()

xpts=data[:,0]
ypts=data[:,1]
# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3)
alldata = np.vstack((xpts, ypts))
fpcs = []

## For showing FPC index for different no of clusters
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('on')
fig1.tight_layout()
plt.show()

## Showing clusters for a single FPC index (max value)
ncenters=2                  # specifying the no of clusters
fpcs = []
fig2, ax = plt.subplots()
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.001, maxiter=1000, init=None)

    # Store fpc values for later
fpcs.append(fpc)
print("Membership value: ",u)
    # Plot assigned clusters, for each data point in training set
cluster_membership = np.argmax(u, axis=0)
print(cluster_membership)
for j in range(ncenters):
    ax.plot(xpts[cluster_membership == j],ypts[cluster_membership == j], '.', color=colors[j],markersize=10)

    # Mark the center of each fuzzy cluster
for pt in cntr:
    ax.plot(pt[0], pt[1], 'ks')         # 'ks'-> black square

ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
ax.axis('on')

# finding total and average probability of all points
total_sum=0
for i in range(len(cluster_membership)):
    total_sum += u[cluster_membership[i]][i]
avg_sum = total_sum / len(cluster_membership)
print("Average Probability:",avg_sum)

##Showing Graph along with labels
plt.title("2 Clusters Using Fuzzy C Means (FCM)")
plt.xlabel("NDVI value")
plt.ylabel("SPAD value")
plt.show()
#fig2.savefig("img/4cluster_fcm_all.png")



################################################################################
##################### ------------- CLASSIFICATION -----------  #################
################################################################################


## Updating the ngrdi data along with their clusters
new_data=updatedata(cluster_membership)

## Distributing data into train and test data
train= new_data.sample(frac=0.7,random_state=1)        # frac->% of data as train data
test= new_data.loc[~new_data.index.isin(train.index)]

## Using Decision Tree Classifier for NGRDI and Cluster
ngrdi_classifier = tree.DecisionTreeClassifier()
## For forcibly assign the max depth of tree, use Below line Instead
#ngrdi_classifier = tree.DecisionTreeClassifier(max_depth=6)

ngrdi_classifier.fit(train[3].values.reshape(len(train),1), train['cluster'].values.reshape(len(train),1))
predictions = ngrdi_classifier.predict(test[3].values.reshape(len(test),1))

## Output the decision tree
#with open("ngrdi_d_tree_fcm1.dot", "w") as f:
#    f = tree.export_graphviz(ngrdi_classifier, out_file=f)

print("Accuracy : ",accuracy_score(test['cluster'].values.reshape(len(test),1),predictions))
#print("Classification Report : ",classification_report(test['cluster'].values.reshape(len(test),1),predictions))

## Prediction Variable, Target Variable and Predictions
print("NGRDI values :",test[3].values)
print("Actual Cluster :",test['cluster'].values)
print("Predicted Cluster :",predictions)
