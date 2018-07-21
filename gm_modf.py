import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import pandas as pd

#Generate data
def gendata():
    obs = pd.read_csv('C:/Users/SUMITKR/Documents/python/gmm/20febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
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
plt.scatter(obs[:,0],obs[:,1],c=labels,s=40,cmap='viridis')
plt.show()
fig.savefig('ndvi_spad_wtout_cluster.png')
probs=ft.predict_proba(obs)
#print(probs.round(5))
