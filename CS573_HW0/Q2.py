import numpy as np
import math
[n,i] = [10000, 0] # n is dimension of X and i is iteration number
x = np.random.randn(n)
[tol,error] = [0.0001,1]# tolerance of error, and error initialization
gama = 10e-6  # step size multiplier
[mu,sigma] = [5,5] # initial value
df_mu = lambda mu, sigma: n*(np.mean(x)-mu)/(sigma**2) # derivative by mu
df_sigma = lambda mu, sigma: -n/sigma+(n*np.var(x))/(sigma**3) # derivative by sigma
while error > tol and i < 10e6:
	[mu,sigma] = map(lambda a,b:a+b,[mu,sigma],\
		[gama*df_mu(mu, sigma), gama*df_sigma(mu, sigma)])
	error = np.linalg.norm([df_mu(mu, sigma), df_sigma(mu, sigma)])
	i = i+1
print([mu,sigma])