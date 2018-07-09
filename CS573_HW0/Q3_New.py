import numpy as np
import math
n=1000
x = np.random.randn(n)
[tol,error] = [0.0001,1].   # tolerance of error, and error initialization
gama = 0.00001              # step size multiplier
[mu,sigma] = [-5,2]         # initial value
i = 0                       # iteration number
df_mu = lambda mu, sigma: n*(np.mean(x)-mu)/(sigma**2)    #
df_sigma = lambda mu, sigma: -n/sigma+(n*np.var(x))/(sigma**3)
while error > tol and i < 500:
	[mu,sigma] = map(lambda a,b:a+b,[mu,sigma],[gama*df_mu(mu, sigma), gama*df_sigma(mu, sigma)])
	error = np.linalg.norm([df_mu(mu, sigma), df_sigma(mu, sigma)])
	i = i+1
print([mu,sigma])