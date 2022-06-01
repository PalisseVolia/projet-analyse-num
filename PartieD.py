
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

eps = math.pow(10,-6)
m = 100
compteur = 0

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

dernierY = np.array([2,2])

Y = dernierY - rho(dernierY)*G(dernierY)
dist = np.linalg.norm(Y-dernierY)
while (compteur <= m and dist > eps):
  compteur += 1
  dernierY = Y
  Y = dernierY - rho(dernierY)*G(dernierY)
  dist = np.linalg.norm(Y-dernierY)
  print(dist)
  
plt.plot(x,err_balayage1,color="r")
print("compteur : ", compteur)
print("Y : ",dernierY)

#-------------------------------- Question 3  ---------------------------------

