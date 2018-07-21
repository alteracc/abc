import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt

#Generate data
def gendata():
    obs = np.concatenate((1.6*np.random.randn(300, 2), 6 + 1.3*np.random.randn(300, 2), np.array([-5, 5]) + 1.3*np.random.randn(200, 2), np.array([2, 7]) + 1.1*np.random.randn(200, 2)))
    return obs

def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return 1/(2*np.pi*xsig*ysig) * np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

#Generate GMM model and fit the data
def gengmm(nc=4, n_iter = 2):
    g = mixture.GMM(n_components=nc)  # number of components
    g.init_params = ""  # No initialization
    g.n_iter = n_iter   # iteration of EM method
    return g

def plotGMM(g, n, pt):
    delta = 0.025
    x = np.arange(-10, 10, delta)
    y = np.arange(-6, 12, delta)
    X, Y = np.meshgrid(x, y)
 
    if pt == 1:
        for i in range(n):
            Z1 = gaussian_2d(X, Y, g.means_[i, 0], g.means_[i, 1], g.covars_[i, 0], g.covars_[i, 1])
            plt.contour(X, Y, Z1, linewidths=0.5)
 
    #print g.means_
    plt.plot(g.means_[0][0],g.means_[0][1], '+', markersize=13, mew=3)
    plt.plot(g.means_[1][0],g.means_[1][1], '+', markersize=13, mew=3)
    plt.plot(g.means_[2][0],g.means_[2][1], '+', markersize=13, mew=3)
    plt.plot(g.means_[3][0],g.means_[3][1], '+', markersize=13, mew=3)

obs = gendata()
fig = plt.figure(1)
print(obs.shape)
g = gengmm(4, 100)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 0)
plt.title('Gaussian Mixture Model')
plt.show()

g = gengmm(4, 1)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 1)')
plt.show()

g = gengmm(4, 5)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 5)')
plt.show()

g = gengmm(4, 20)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 20)')
plt.show()

g = gengmm(4, 100)
g.fit(obs)
plt.plot(obs[:, 0], obs[:, 1], '.', markersize=3)
plotGMM(g, 4, 1)
plt.title('Gaussian Models (Iter = 100)')
plt.show()
