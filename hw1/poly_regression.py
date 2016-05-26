'''Implement polynomial curve fitting in python'''

import numpy as np
import matplotlib.pyplot as plt 

def genData(samples):
	X = np.arange(-np.pi, np.pi, 2*np.pi*1.0/samples)
	y = np.sin(X)
	mu = 0
	sigma = 0.1
	noise = np.random.normal(mu,sigma,len(y))
	y = y + noise
	return X,y

def fitCurve(X,y,order = 0,regular = 0):
	A = []
	for x_i in X:
		A_i = []
		for j in range(0, order + 1):
			A_i.append(np.power(x_i, j))
		A.append(A_i)
	A = np.array(A)
	y = np.array(y)

	# W = (A^T*A)^(-1)*A^T*y (no regular)
	# W = (A^T*A + lambda*I)^(-1)*A^T*y (regularization)
	W = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + np.eye(order + 1, order + 1)*regular), A.T), y)

	X_orgin = np.arange(-np.pi, np.pi, 0.01)
	y_orgin = []
	for x_i in X_orgin:
		y_orgin.append(np.sin(x_i))

	X_fit = np.arange(-np.pi, np.pi, 0.01)
	y_fit = []
	for x_i in X_fit:
		fit = 0
		for j in range(0, order + 1):
			fit = fit + np.power(x_i, j)*W[j]
		y_fit.append(fit)
	return X_orgin,y_orgin,X_fit,y_fit

if __name__ == '__main__':
	
	# 1, y = sin(x) with gaussian noise, where noise ~ N(0,0.1)
	X,y = genData(10)
	plt.scatter(X, y, 35.0, color = 'b', label = 'noise')
	X_orgin,y_orgin,X_fit,y_fit = fitCurve(X,y)
	plt.plot(X_orgin, y_orgin, color = 'g', linewidth = 2, label = 'y = sin(x)')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.axis([-np.pi-0.1,np.pi+0.1, -2,2])
	plt.show()

	# 2. fit degree 3 in 10 samples
	X,y = genData(10)
	plt.scatter(X, y, 35.0, color = 'b', label = 'noise')
	X_orgin,y_orgin,X_fit,y_fit = fitCurve(X,y,3)
	plt.plot(X_orgin, y_orgin, color = 'g', linewidth = 2, label = 'y = sin(x)')
	plt.plot(X_fit, y_fit, color = 'r', linewidth = 2, label = 'M = 3')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.axis([-np.pi-0.1,np.pi+0.1, -2,2])
	plt.show()

	# 3. fit degree 9 in 10 examples
	X,y = genData(10)
	plt.scatter(X, y, 35.0, color = 'b', label = 'noise')
	X_orgin,y_orgin,X_fit,y_fit = fitCurve(X,y,9)
	plt.plot(X_orgin, y_orgin, color = 'g', linewidth = 2, label = 'y = sin(x)')
	plt.plot(X_fit, y_fit, color = 'r', linewidth = 2, label = 'M = 9')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.axis([-np.pi-0.1,np.pi+0.1, -2,2])
	plt.show()

	# 4. fit degree 9 in 15 examples
	X,y = genData(15)
	plt.scatter(X, y, 35.0, color = 'b', label = 'noise')
	X_orgin,y_orgin,X_fit,y_fit = fitCurve(X,y,9)
	plt.plot(X_orgin, y_orgin, color = 'g', linewidth = 2, label = 'y = sin(x)')
	plt.plot(X_fit, y_fit, color = 'r', linewidth = 2, label = 'M = 9')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.axis([-np.pi-0.1,np.pi+0.1, -2,2])
	plt.show()

	# 5. fit degree 9 in 100 examples
	X,y = genData(100)
	plt.scatter(X, y, 35.0, color = 'b', label = 'noise')
	X_orgin,y_orgin,X_fit,y_fit = fitCurve(X,y,9)
	plt.plot(X_orgin, y_orgin, color = 'g', linewidth = 2, label = 'y = sin(x)')
	plt.plot(X_fit, y_fit, color = 'r', linewidth = 2, label = 'M = 9')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.axis([-np.pi-0.1,np.pi+0.1, -2,2])
	plt.show()

	# 6. fit degree 9 in 10 examples with regularization ln(lambda) = -18
	X,y = genData(10)
	plt.scatter(X, y, 35.0, color = 'b', label = 'noise')
	X_orgin,y_orgin,X_fit,y_fit = fitCurve(X,y,9,np.exp(-3))
	plt.plot(X_orgin, y_orgin, color = 'g', linewidth = 2, label = 'y = sin(x)')
	plt.plot(X_fit, y_fit, color = 'r', linewidth = 2, label = '$\lambda$ = -3')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.axis([-np.pi-0.1,np.pi+0.1, -2,2])
	plt.show()
