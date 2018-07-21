import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import pandas as pd

#Generate data
def gendata():
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
def gengmm(nc=4, n_iter = 2):
    g = mixture.GMM(n_components=nc)  # number of components
    g.init_params = ""  # No initialization
    g.n_iter = n_iter   # iteration of EM method
    return g

obs = gendata()

n_components = np.arange(1, 16)
BIC = np.zeros(n_components.shape)
AIC = np.zeros(n_components.shape)

for i, n in enumerate(n_components):
    clf = mixture.GMM(n_components=n,
              covariance_type='diag')
    clf.fit(obs)

    AIC[i] = clf.aic(obs)
    BIC[i] = clf.bic(obs)

fig=plt.figure()
#plt.plot(n_components, AIC, label='AIC')
plt.plot(n_components, BIC, label='BIC')
plt.legend(loc=0)
plt.xlabel('No of Clusters')
plt.ylabel('Bayesian Information Criterion')
#plt.ylabel('AIC / BIC')
plt.title("BIC vs Number of Clusters")
#plt.tight_layout()
plt.show()
fig.savefig('BIC_vs_no_of_cluster_graph.png')

i_n = np.argmin(BIC)

clf = mixture.GMM(n_components[i_n])
clf.fit(obs)
label = clf.predict(obs)
print("Score: ",clf.score(obs))
plt.figure()
plt.scatter(obs[:, 0], obs[:, 1], c=label, s=16, lw=0)
plt.title('classification at min(BIC)')
plt.show()
