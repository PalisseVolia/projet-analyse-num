#%%
from turtle import color
from matplotlib import colors
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#------------------------------------------------------------------------------
#         E.   Application à des problèmes de transfert de chaleur
#------------------------------------------------------------------------------

#-------------------------------- Question 1  ---------------------------------

def G(y):
    return 2*(np.matmul(A,y)-b)

def rho(y):
	if np.array_equal(G(y),[0,0]):
		return 0
	else:
		return ((np.linalg.norm(G(y)))**2)/(2*np.matmul(np.transpose(G(y)),np.matmul(A,G(y))))

def solveurg(A, b,t):             # A matrice carrée de taille n, b les n résutats des équantions du système d'équations
    m = 500
    n = len(A)
    eps = 10**-6
    Yn = 3*np.ones(n)           # Initialisation de Yn avec des valeurs arbitraires (ici 3 pour toutes les composantes))
    nb_itérations = 0
    Y = Yn - rho(Yn)*G(Yn)
    while (nb_itérations <= m and np.linalg.norm(Y-Yn) > eps):
        nb_itérations += 1
        Yn = Y
        Y = Yn - rho(Yn)*G(Yn)
    x = np.linspace(0,1,n)
    fig1 = plt.figure(figsize = (15,10))
    plt.plot(x,Y)
    plt.title(t ,fontsize = 13)
    return Yn

n = 20
A = np.zeros((n,n))
A = np.diag(2*np.ones(n)) + np.diag(-1*np.ones(n-1),1) + np.diag(-1*np.ones(n-1),-1)
b = np.zeros(n)
b[0] = 500
b[n-1] = 350

print(A)
print(b)
print(solveurg(A,b,"Evolution de la température pour la paroi calorifugée"))

#-------------------------------- Question 2  ---------------------------------
n = 20
A = np.zeros((n,n))
A = np.diag(2*np.ones(n)) + np.diag(-1*np.ones(n-1),1) + np.diag(-1*np.ones(n-1),-1)
dx = 1/(n+1)
D = np.sqrt(0.1)
a = 500
b = 350
Ta = 300
c = (1/D)**2
C = c * np.diag(np.ones(n))
f = c*Ta
B = np.zeros(n)
for i in range(len(B)):
    B[i] = (dx**2) * f
B[0] = (dx**2) * f + a
B[n-1] = (dx**2) * f + b
print(B)
Xn = A + dx**2 * C 
print(solveurg(Xn,B,"Evolution de la température pour la paroi non calorifugée" ))


# %%
