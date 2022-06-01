#%%

from turtle import color
from matplotlib import colors
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#------------------------------------------------------------------------------
#         D.   Calcul de l'inverse d'une matrice
#------------------------------------------------------------------------------

#-------------------------------- Question 1  ---------------------------------

# Voir le rapport

#-------------------------------- Question 2  ---------------------------------

eps = 10**-6
m = 100
nb_itérations = 0

b = np.array([-3,5])
A = np.array([[9,0],[0,9]])

def G(y):
	return 2*(np.matmul(A,y)-b)

def rho(y):
	if np.array_equal(G(y),[0,0]):
		return 0
	else:
		num = (np.linalg.norm(G(y)))**2
		den = 2*np.matmul(np.transpose(G(y)),np.matmul(A,G(y)))
		return num/den

Yn = np.array([2,2])

Y = Yn - rho(Yn)*G(Yn)
dist = np.linalg.norm(Y-Yn)
while (nb_itérations <= m and dist > eps):
	nb_itérations += 1
	Yn = Y
	Y = Yn - rho(Yn)*G(Yn)
	dist = np.linalg.norm(Y-Yn)
	print(dist)
  
plt.plot(x,err_balayage1,color="r")
print("nb_itérations : ", nb_itérations)
print("Y : ",Yn)

#-------------------------------- Question 3  ---------------------------------

