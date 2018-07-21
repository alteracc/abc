import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import pandas as pd

#Generate data
def gendata():
    obs_2=pd.read_csv('30Jancluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_3=pd.read_csv('06febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_4=pd.read_csv('13febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_5=pd.read_csv('20febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_6=pd.read_csv('28febcluster.csv',usecols=[1,4],header=None,skiprows=[0])
    obs_7=pd.read_csv('6marCluster.csv',usecols=[1,4],header=None,skiprows=[0])
    frames=[obs_2,obs_3,obs_4,obs_5,obs_6,obs_7]
    obs=pd.concat(frames,ignore_index=True)
    return obs.values

def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return 1/(2*np.pi*xsig*ysig) * np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

#Generate GMM model and fit the data
def gengmm(nc=4, n_iter = 2):
    g = mixture.GMM(n_components=nc)  # number of components
    g.init_params = ""  # No initialization
    g.n_iter = n_iter   # iteration of EM method
    return g

def plotGMM(g, n, pt):
    delta = 0.05
    x = np.arange(0.1, 0.9,delta)
    y = np.arange(20, 60,delta)
    X, Y = np.meshgrid(x, y)
 
    if pt == 1:
        #plot the GMM with mixing parameters (weights)
        #i=0
        #Z2= g.weights_[i]*gaussian_2d(X, Y, g.means_[i, 0], g.means_[i, 1], g.covars_[i, 0], g.covars_[i, 1])
        #for i in xrange(1,n):
        #    Z2 = Z2+ g.weights_[i]*gaussian_2d(X, Y, g.means_[i, 0], g.means_[i, 1], g.covars_[i, 0], g.covars_[i, 1])
        #plt.contour(X, Y, Z2)
        
        for i in range(n):
            Z1 = gaussian_2d(X, Y, g.means_[i, 0], g.means_[i, 1], g.covars_[i, 0], g.covars_[i, 1])
            plt.contour(X, Y, Z1, linewidths=1)
 
    #print g.means_
    for j in range(n):
        plt.plot(g.means_[j][0],g.means_[j][1], '+', markersize=13, mew=3)

obs = gendata()
fig = plt.figure(1)
g = gengmm(4, 100)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 0)
plt.title('Gaussian Mixture Model')
plt.show()
fig.savefig('gmm1_all.png')

fig = plt.figure(1)
g = gengmm(4, 1)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 1)')
plt.show()
fig.savefig('gmm2_all.png')

fig = plt.figure(1)
g = gengmm(4, 5)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 5)')
plt.show()
fig.savefig('gmm3_all.png')

fig = plt.figure(1)
g = gengmm(4, 20)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 20)')
plt.show()
fig.savefig('gmm4_all.png')

fig = plt.figure(1)
g = gengmm(4, 100)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 100)')
plt.show()
fig.savefig('gmm5_all.png')
