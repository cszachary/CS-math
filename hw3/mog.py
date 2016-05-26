'''Implement K-means and MOG in D-dimension case'''
import matplotlib.pyplot as plt
import numpy as np 
import random
from scipy.linalg import norm
import numpy.matlib as ml 

def genData():
	
	plt.figure(1)

	mean1 = [-10,-5]
	cov1 = [[20,0],[10,15]]
	x1,y1 = np.random.multivariate_normal(mean1,cov1,200).T
	plt.plot(x1,y1,'o')

	mean2 = [0,5]
	cov2 = [[1,0],[0,1]]
	x2,y2 = np.random.multivariate_normal(mean2,cov2,200).T
	plt.plot(x2,y2,'o')
 
	mean3 = [10,-5]
	cov3 = [[2,0],[0,2]]
	x3,y3 = np.random.multivariate_normal(mean3,cov3,200).T
	plt.plot(x3,y3,'o')

	plt.axis('equal')
	plt.show()

	X = np.zeros((600,2))
	y = np.zeros((600,1))

	for i in range(200):
		X[i][0] = x1[i]
		X[i][1] = y1[i]
		y[i] = 1
	for j in range(200):
		X[i+j][0] = x2[j]
		X[i+j][1] = y2[j]
		y[i+j] = 2
	for k in range(200):
		X[i+j+k][0] = x3[k]
		X[i+j+k][1] = y3[k]
		y[i+j+k] = 3

	return X,y 

def distMat(X,Y):
	n = len(X)
	m = len(Y)
	xx = ml.sum(X*X,axis = 1)
	yy = ml.sum(Y*Y,axis = 1)
	xy = ml.dot(X,Y.T)

	return np.tile(xx,(m,1)).T + np.tile(yy,(n,1))-2*xy

def Kmeans(X,k,observer=None,threshold=1e-15,maxiter=300):

	N = len(X)
	labels = np.zeros(N,dtype=int)
	centers = np.array(random.sample(X,k))
	iter = 0

	def calc_J():
		sum = 0
		for i in xrange(N):
			sum += norm(X[i] - centers[labels[i]])
		return sum

	Jprev = calc_J()
	while True:
		if observer is not None:
			observer(iter,labels,centers)

		dist = distMat(X,centers)
		labels = dist.argmin(axis = 1)
		for j in range(k):
			idx_j = (labels==j).nonzero()
			centers[j] = X[idx_j].mean(axis = 0)

		J = calc_J()
		iter += 1

		if Jprev - J<threshold:
			break
		Jprev = J

		if iter>=maxiter:
			break

		if observer is not None:
			observer(iter,labels,centers)

def gmm(X,K,observer = None, threshold = 1e-15, maxiter = 10000):
	# X: N-by-D matrix
	# K: K is the number of components
	# PX: N-by-K matrix indicating the probability of each component generating each point
	# Mu: K-by-D matrix 
	# Sigma: D-by-D-by-K matrix
	# Pi: 1-by-K matrix
	iter = 0
	(N,D) = X.shape
	# random init K centroids
	centroids = np.array(random.sample(X,K))
	labels = np.zeros(N,dtype = int)

	def init_params():
		Mu = centroids
		Pi = np.zeros((1,K))
		Sigma = np.zeros((D,D,K))

		dist = distMat(X,Mu)
		labels = dist.argmin(axis = 1)
		for k_i in range(K):
			idx_k = (labels == k_i).nonzero()
			X_ki = X[idx_k]
			Pi[0,k_i] = len(idx_k)/(N*1.0)
			Sigma[:,:,k_i] = np.cov(X_ki.T)

		return Mu,Pi,Sigma

	def calc_prob():
		PX = np.zeros((N,K))
		for k_i in range(K):
			xshift = X - np.tile(Mu[k_i,:],(N,1))
			inv_Sigma = np.linalg.inv(Sigma[:,:,k_i])
			tmp = np.sum(np.dot(xshift,inv_Sigma)*xshift,axis = 1)
			coef = np.power((2*np.pi),-D/2)*np.sqrt(np.linalg.det(inv_Sigma))
			PX[:,k_i] = coef * np.exp(-0.5*tmp)

		return PX

	Mu,Pi,Sigma = init_params()
	Lprev = -np.inf

	# EM algorithm
	while True:
		if observer is not None:
			observer(iter,labels,centroids,1)

		# estimation step
		PX = calc_prob()
		labels = PX.argmax(axis = 1)

		pGamma = PX*np.tile(Pi,(N,1))
		pGamma = pGamma/np.tile(np.sum(pGamma,axis = 1),(K,1)).T

		Nk = np.sum(pGamma,axis = 0)
		
		Mu = np.dot(np.dot(np.diag(1.0/Nk),pGamma.T),X)
		Pi = Nk/N

		for k_i in range(K):
			xshift = X - np.tile(Mu[k_i,:],(N,1))
			Sigma[:,:,k_i] = np.dot(np.dot(xshift.T,np.diag(pGamma[:,k_i])),xshift)/Nk[k_i]
		
		L = np.sum(np.log(np.dot(PX,Pi.T)))
		if L - Lprev < threshold:
			break
		
		Lprev = L
		iter += 1

		if iter>=maxiter:
			break

		if observer is not None:
			observer(iter,labels,centroids,1)

	return PX,Mu,Sigma,Pi

if __name__ =='__main__':
	
	X,y = genData()
	
	def observer(iter,labels,centers,flag = 0):
	
		colors = np.array([[1,0,0],[0,1,0],[0,0,1]])
		plt.plot(hold = False)
		plt.hold(True)

		data_color = [colors[c] for c in labels]
		plt.scatter(X[:,0],X[:,1],c=data_color,alpha= 0.5)
		plt.scatter(centers[:,0],centers[:,1],s = 150,c = colors)
		if flag:
			plt.savefig("GMM_%diter.png" %iter)
		else:
			plt.savefig("KMeans_%diter.png" %iter)
	Kmeans(X,3,observer=observer)
	gmm(X,3,observer = observer)


