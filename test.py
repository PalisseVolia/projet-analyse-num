#%%
#Question 2

import math
import numpy as np
import matplotlib.pyplot as plt

eps = math.pow(10,-6)
n = 100
m = 10000000
compteur = 0
c = 10
f = 3000

def tridiag(N):
  mat = np.diag(np.ones(N),0)*2 - np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1)
  return mat


Y1 = np.zeros(n) #tableau contenant les valeurs de la dérivée seconde de la température en chaque point du schéma de différence finie
A = ((n+1)**2)*tridiag(n)
A+= c*np.diag(np.ones(n),0)
b = f*np.ones(n)  #les b(i) sont les approximations de la valeurs de la température au point xi
b[0] += 500*((n+1)**2)  #on connaît la valeur en 0 (b(0)) et celle en 1 (b(n-1))
b[n-1] += 350*((n+1)**2)


def G(y):
  return 2*(np.matmul(A,y)-b)

def rho(y):
  if np.array_equal(G(y),[0,0]):
    return 0
  else:
     res = ((np.linalg.norm(G(y)))**2)/(2*np.matmul(np.transpose(G(y)),np.matmul(A,G(y))))
     return res


Y2 = Y1 - rho(Y1)*G(Y1)
dist = np.linalg.norm(Y2-Y1)
while (compteur <= m and dist > eps):
  compteur += 1
  Y1 = Y2
  Y2 = Y1 - rho(Y1)*G(Y1)
  dist = np.linalg.norm(Y2-Y1)

x= np.linspace(0,1,n)
plt.plot(x,Y1,color="r")

print("compteur : ", compteur)
# %%
