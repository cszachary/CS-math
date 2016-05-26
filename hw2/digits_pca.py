'''Implement PCA of digits 3 in 2D'''
from __future__ import division
from numpy import linalg as la
from numpy import *
import matplotlib
import matplotlib.pyplot as plt 
import sys

global skiplines
skiplines = 21
global num
num = 1934
global row 
row = 32
global column
column = 32

def img2vector(filename):
	imgVect = zeros((num,1024))
	labelVect = zeros((num,1))
	fr = open(filename)
	for i in range(skiplines):
		invalid = fr.readline()
	for k in range(num):
		for i in range(row):
			lineStr = fr.readline()
			for j in range(column):
				imgVect[k,row*i+j] = int(lineStr[j])
		labelStr = fr.readline()
		labelVect[k,0] = int(labelStr[1])
	return imgVect,labelVect

def autoNorm(data):
	mean = sum(data,axis=0)
	m = data.shape[0]
	mean = mean/m
	normData = data - tile(mean,(m,1))
	return normData

def pickPoints(data):
	m = data.shape[0]
	result = list()
	for y in range(-4,6,2):
		for x in range(-4,6,2):
			minVals = sys.maxint
			for k in range(m):
				s = (data[k][0]-x)**2+(data[k][1]-y)**2
				if s<minVals:
					minVals = s
					idx = k
			result.append(int(idx))
	return result

def pca(data):
	m = data.shape[0]
	X = 1/m * dot(data.T,data)
	U,S,V = la.svd(X)
	X_pca = dot(data,U[:,0:2])
	X_recv = dot(X_pca,U[:,0:2].T)
	result = pickPoints(X_pca)
	x = X_pca[:,0]
	y = X_pca[:,1]
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	ax.scatter(x,y,5.0,label="Projected Points")
	for i in result:
		x_pick = X_pca[i,0]
		y_pick = X_pca[i,1]
		ax.scatter(x_pick,y_pick,marker='o',color='r')
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.grid()
	plt.legend()
	return X_recv,result

def pcaRecover(data,index):
	X_show = zeros((25,1024))
	for i in range(len(index)):
		X_show[i,:] = data[index[i],:]
	rows = columns = 5
	heights = widths = 32
	pad = 1
	m = 25
	cur = 1
	display_array = ones((pad+rows*(heights+pad),pad+columns*(widths+pad)))
	for j in range(1,rows+1):
		for i in range(1,columns+1):
			if cur>m:
				break
			maxVal = max(abs(X_show[cur-1,:]))
			row_start = pad+(j-1)*(heights+pad)+1
			row_end = pad+(j-1)*(heights+pad)+heights+1
			col_start = pad+(i-1)*(widths+pad)+1
			col_end = pad+(i-1)*(widths+pad)+widths+1
			display_array[row_start:row_end,col_start:col_end]=reshape(X_show[cur-1,:],(widths,heights))/maxVal
			cur = cur+1
		if cur>m:
			break
	plt.figure(2)
	plt.imshow(display_array,cmap=matplotlib.cm.gray)


if __name__=='__main__':
	filename = "optdigits-orig.tra"
	imgVect,labelVect = img2vector(filename)
	idx3 = (labelVect[:,0]==3)
	digit3Vect = imgVect[idx3,:]
	normVect = autoNorm(digit3Vect)
	X_recv,index = pca(normVect)
	pcaRecover(X_recv,index)
	plt.show()