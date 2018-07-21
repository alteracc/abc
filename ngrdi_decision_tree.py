import math
import numpy as np
from sklearn import mixture
from sklearn.cross_validation import train_test_split
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
import pydotplus
import collections

#Generate data
def gendata():
    ## COL(indexing from 0) 1 & 4 --> contains NDVI and SPAD respectively
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

#Generate GMM model and fit the data
def gengmm(nc, n_iter):
    g = mixture.GMM(n_components=nc)  # number of components
    g.init_params = ""  # No initialization
    g.n_iter = n_iter   # iteration of EM method
    return g

def update_ngrdi(g,labels):
    ## COL(indexing from 0) 3 -->  contains NGRDI values
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
    new_obs=obs
#    new_obs['cx']=[g.means_[i][0] for i in labels]
#    new_obs['cy']=[g.means_[i][1] for i in labels]
    new_obs['cluster']=[i for i in labels]
    print(new_obs)
    return new_obs

################################################################################
#################------------ CLUSTERING ---------------########################
################################################################################


obs = gendata()
ncenters=2
g = gengmm(ncenters, 1000)
ft=g.fit(obs)

## Clusters of all points
labels=ft.predict(obs)
print(labels)
fig=plt.figure()

probs=ft.predict_proba(obs)

#calculating the total and average probability of points
total_prob=0
for i in range(len(obs)):
    total_prob += probs[i][labels[i]]
avg_prob = total_prob / len(obs)
print("Average Probability: ",avg_prob)

## Plotting points size according to the probability in the cluster
size = 50 * probs.max(1) ** 2

## Showing Clusters of Data
plt.scatter(obs[:,0],obs[:,1],c=labels,s=size,cmap='viridis')
plt.title('2 Clusters Using Gaussian Mixture Model (GMM)')
plt.xlabel("NDVI value")
plt.ylabel("SPAD value")
#legend_field="Avg Probability = "+str(avg_prob)[0:5]
#plt.text(0.1,58,legend_field,fontdict=None)
plt.show()
## Saving image
fig.savefig('img/2cluster_gmm_all.png')

## Cluster Centroid Locations
for i in range(ncenters):
    print("Cluster-",(i)," -> Centroid : ",g.means_[i])

## For printing probability of all data in a dataset
#np.set_printoptions(threshold=np.inf)
#print(probs.round(5))



################################################################################
################ ---------------- CLASSIFICATION -------------- ####################
################################################################################



#######  Using Decision Tree Clasifier for NGRDI and Cluster

data=update_ngrdi(g,labels)

train= data.sample(frac=0.9,random_state=10)
test= data.loc[~data.index.isin(train.index)]

ngrdi_classifier = tree.DecisionTreeClassifier()

## For forcibly assign the max depth of tree, use Below line Instead
#ngrdi_classifier = tree.DecisionTreeClassifier(max_depth=6)

ngrdi_classifier.fit(train[3].values.reshape(len(train),1), train['cluster'].values.reshape(len(train),1))

predictions = ngrdi_classifier.predict(test[3].values.reshape(len(test),1))

#dot_data = tree.export_graphviz(ngrdi_classifier,feature_names=['3'],out_file=None,filled=True,rounded=True)
#graph = pydotplus.graph_from_dot_data(dot_data)
#
#colors = ('turquoise', 'orange')
#edges = collections.defaultdict(list)
#
#for edge in graph.get_edge_list():
#    edges[edge.get_source()].append(int(edge.get_destination()))
#
#for edge in edges:
#    edges[edge].sort()    
#    for i in range(2):
#        dest = graph.get_node(str(edges[edge][i]))[0]
#        dest.set_fillcolor(colors[i])
#
#graph.write_png('ngrdi_tree.png')


## Output the decision tree
#with open("ngrdi_d_tree6.dot", "w") as f:
#    f = tree.export_graphviz(ngrdi_classifier, out_file=f)

print("Accuracy : ",accuracy_score(test['cluster'].values.reshape(len(test),1),predictions))
#print("Classification Report : ",classification_report(test['cluster'].values.reshape(len(test),1),predictions))

print("NGRDI values :", test[3].values)
print("Actual Cluster :",test['cluster'].values)
print("Predicted Cluster :",predictions)
