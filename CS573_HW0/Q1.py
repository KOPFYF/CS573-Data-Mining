import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

print('nano')
X1 = np.random.randn(1000) #normal
X2 = np.random.randn(1000)
plt.plot(X1, X2, 'ro')
plt.axis([-2, 2, -2, 2])
plt.xlabel('X1')
plt.ylabel('X2')
plt.savefig("Q1(b)_n1000.png")

R = X1**2 + X2**2
r = [i for i in R if i <= 0.5**2]
p_b = len(r)/1000
print(p_b)

X3 = np.random.randn(1000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, X3)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.savefig("Q1(c)_n1000.png")

R3 = R + X3**2
r3 = [i for i in R3 if i <= 0.5**2]
p_c = len(r3)/1000
print(p_c)

plt.show()
