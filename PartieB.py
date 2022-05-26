#%%
from turtle import color
from matplotlib import colors
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#------------------------------------------------------------------------------
#         B.   Optimisation d'une fonction d'une variable réelle
#------------------------------------------------------------------------------

print("B.   Optimisation d'une fonction d'une variable réelle")
print()

#-------------------------------- Question 1  ---------------------------------

def balayage_constant(func,a,b,N):
    dx = (b-a)/N
    ai = a
    bi = a+dx
    minimum = func(a)
    for i in range(N):
    #Pour les N intervalles
        for j in (ai,bi):
        #On cherche les minimums dans les sous-intervalles
            if (func(j)<minimum):
                minimum=func(j)
        #on modifie les sous-intervalles
        ai=bi
        bi=bi+dx
    return minimum


def balayage_aleatoire(func,a,b,N):
    val = np.zeros(N+1)
    for i in range(len(val)):
    #On remplit un tableau de valeurs aléatoires
        r=random.uniform(a,b) #random.uniform permet de renvoyer des floats
        #aléatoires
        val[i]=r
    minimum = func(val[0])
    for i in val:
    #On cherche la valeur minimum en utilisant les nombres aléatoires comme
    #antécédents de la fonction
        if func(i)<minimum:
            minimum = func(i)
    return minimum

#-------------------------------- Question 2  ---------------------------------

def func(x):
    return x**3 - 3*x**2 + 2*x + 5

def dfunc(x):
#Dérivée de f(x)
    return 3*x**2 - 6*x + 2

def droite(x):
    return 0

def graph_2D(func,a,b):
#Graph de la fonction
    x = [k for k in np.arange(a,b,0.001)]
    result = []
    for i in np.arange(a,b,0.001):
        result.append(func(i))
    plt.plot(x,result)
    
minimum_reel = 4.615099820540249
#= 5 - 2 / (3 * sqrt(3))

print("Question 2 ") 

print("La valeur réelle minimum = ")
print(minimum_reel)

print("Par balayage constant, on obtient minimum = ")
print(balayage_constant(func, 0, 3,100))

print("Par balayage aléatoire, on obtient minimum = ")
print(balayage_aleatoire(func, 0, 3,100))
print()

#graph_2D(func,0,3)
#graph_2D(dfunc,0,3)
#graph_2D(droite,0,3)

#-------------------------------- Question 3  ---------------------------------
a = 0
b = 3
size = 30

errbalayconst = np.zeros(size)
for i in range(0,size):
    errbalayconst[i] = abs(balayage_constant(func,a,b,i+1)-minimum_reel)

errbalayalea = np.zeros(size)
for i in range(0,size):
    errbalayalea[i] = abs(balayage_aleatoire(func,a,b,i+1)-minimum_reel)
    
x = [k for k in range(0,size)]
plt.plot(x,errbalayconst, color = 'red')
plt.plot(x,errbalayalea, color = 'blue')
plt.legend(["Balayage constant","Balayage aléatoire"])

#-------------------------------- Question 4  ---------------------------------
def balayage_constant_max(func,a,b,N):
    dx = (b-a)/N
    ai = a
    bi = a+dx
    maximum = func(a)
    for i in range(N):
    #Pour les N intervalles
        for j in (ai,bi):
        #On cherche les minimums dans les sous-intervalles
            if (func(j)>maximum):
                maximum = func(j)
        #on modifie les sous-intervalles
        ai=bi
        bi=bi+dx
    return maximum

print("Question 4 ")
print("Par balayage constant, on obtient maximum = ")
print(balayage_constant_max(func, 0, 3,100))
print()
#-------------------------------- Question 5  ---------------------------------

#Sur papier
#Si fn est croissante, f' est positive et uf' est donc négatif, Xn+1 est plsu petit que Xn, mm raisonnement pour f décroissante (Xn)n se stabilise donc dans les minimums locaux

#-------------------------------- Question 6  ---------------------------------

def gradient_1D(func,dfunc,xn,u,n):
    
    for i in range(1,n):
        xn = xn + u * dfunc(xn)
    minimum = func(xn) 
    return minimum

print("Question 6 ")
print("Par gradient 1D, on obtient minimum = ")
print(gradient_1D(func, dfunc, 0.5, -0.1, 100))
print()


#-------------------------------- Question 7  ---------------------------------

#Sur papier

# %%
