'''Implement L-M algorithm with sympy;
   Need install sympy package;
   use command: pip install sympy
'''

import sympy as sp

def preprocessing(f_str, symbols_str):
	f = sp.simplify(f_str)
	symbol = sp.symbols(symbols_str, seq=True)
	return [f, symbol]

def allgtzero(dictionary):
	for k in dictionary.iterkeys():
		if k <= 0:
			return False
	return True

def lm(f_str, symbols_str):
	[f, symbols] = preprocessing(f_str, symbols_str)
	# # of variables
	D = len(symbols)
	hess = sp.hessian(f, symbols)
	mexpr = sp.Matrix([f])
	grad = mexpr.jacobian(symbols)
	grad = grad.T
	# initial value
	xk = sp.zeros(D, 1)
	fk = 0
	uk = 0.000000001
	epsilon = 0.00001
	for k in range(100):
		Xk_map = {} 
		for i in range(D):
			Xk_map[symbols[i]] = xk[i, 0]
		fk = f.evalf(subs=Xk_map)
		gk = grad.evalf(subs=Xk_map)
		Gk = hess.evalf(subs=Xk_map)
		if gk.norm() < epsilon:
			break
		while  True:
			eigvalue = sp.Matrix.eigenvals(Gk + uk * sp.eye(D))
			if allgtzero(eigvalue):
				break
			else:
				uk = uk * 4
# 		sk = sp.(A=Gk + uk * sp.ones(D), b=-gk)
		sk = (Gk + uk * sp.eye(D)).LUsolve(-gk)
		Xsk_map = {}
		for i in range(D):
			Xsk_map[symbols[i]] = xk[i, 0] + sk[i, 0]
		fxsk = f.evalf(subs=Xsk_map)
		delta_fk = fk - fxsk
		
		t1 = (gk.T * sk)[0, 0]
		t2 = (0.5 * sk.T * Gk * sk)[0, 0]
		qk = fk + t1 + t2
		delta_qk = fk - qk
		rk = delta_fk / delta_qk
		if rk < 0.25:
			uk = uk * 4
		elif rk > 0.75:
			uk = uk / 2
		else:
			uk = uk
		
		if rk <= 0:
			xk = xk
		else:
			xk = xk + sk
		
	print f, symbols
	for i in range(D):
		print symbols[i], ' = ', xk[i]
	print 'min f =', fk
	return [xk, fk, k]

if __name__=='__main__':
	[var, f_value, iters] = lm('cos(exp(x))', 'x')

	[var, f_value, iters] = lm('(x-1)**2+(exp(y)-2)**2', "x y")

	[var, f_value, iters] = lm('(x+y+2)**2+(x-y+1)**2', 'x y')
