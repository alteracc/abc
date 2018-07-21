import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import pandas as pd

#Generate data
def gendata():
    obs_1=pd.read_csv('19Jancluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_2=pd.read_csv('30Jancluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_3=pd.read_csv('06febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_4=pd.read_csv('13febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_5=pd.read_csv('20febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_6=pd.read_csv('28febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_7=pd.read_csv('6marCluster.csv',usecols=[1,4],header=None,skiprows=[0])
    frames=[obs_1,obs_2,obs_3,obs_4,obs_5,obs_6,obs_7]
    obs=pd.concat(frames,ignore_index=True)
    return obs.values

#Generate GMM model and fit the data
def gengmm(nc=4, n_iter = 2):
    g = mixture.GMM(n_components=nc)  # number of components
    g.init_params = ""  # No initialization
    g.n_iter = n_iter   # iteration of EM method
    return g

obs = gendata()
g = gengmm(4, 100)
ft=g.fit(obs)
labels=ft.predict(obs)
print(labels)
fig=plt.figure()
probs=ft.predict_proba(obs)
size = 50 * probs.max(1) ** 2
plt.scatter(obs[:,0],obs[:,1],c=labels,s=size,cmap='viridis')
plt.show()
for i in range(4):
    print("Cluster-",(i)," -> Centroid : ",g.means_[i])
min_spad=[1000 for i in range(4)]  # initializing with out of range value for spad
max_spad=[obs[0][0] for i in range(4)]
max_ndvi=[obs[0][0] for i in range(4)]
min_ndvi=[1000 for i in range(4)] # initializing with out of range value for ndvi
for j in range(len(labels)):
    if(min_spad[labels[j]]>obs[j][1]):
        min_spad[labels[j]]=obs[j][1]
    if(min_ndvi[labels[j]]>obs[j][0]):
        min_ndvi[labels[j]]=obs[j][0]
    if(max_spad[labels[j]]<obs[j][1]):
        max_spad[labels[j]]=obs[j][1]
    if(max_ndvi[labels[j]]<obs[j][0]):
        max_ndvi[labels[j]]=obs[j][0]
print('Cluster','Min-SPAD','Max-SPAD','Min-NDVI','Max-NDVI')
for k in range(4):
    print(k,min_spad[k],max_spad[k],min_ndvi[k],max_ndvi[k])
#fig.savefig('ndvi_spad_cluster_all_wt_size.png')
#np.set_printoptions(threshold=np.inf)
#print(probs.round(5))
