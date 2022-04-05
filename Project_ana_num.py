#------------------------------------------------------------------------------
#        
#                        Projet d'analyse numérique
#
#------------------------------------------------------------------------------
#%%

from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#------------------------------------------------------------------------------
#         B.   Optimisation d'une fonction d'une variable réelle
#------------------------------------------------------------------------------


#-------------------------------- Question 1  ---------------------------------

minbalayconst = []
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
                minbalayconst.append(minimum)
                minimum=func(j)
        #on modifie les sous-intervalles
        ai=bi
        bi=bi+dx
    return minimum

minbalayalea = []
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
            minbalayalea.append(minimum)
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
print(balayage_constant(func, 0, 3,1000))

print("Par balayage aléatoire, on obtient minimum = ")
print(balayage_aleatoire(func, 0, 3,1000))
print()

graph_2D(func,0,3)
graph_2D(dfunc,0,3)
graph_2D(droite,0,3)

plt.close()
#-------------------------------- Question 3  ---------------------------------
a = 0
b = 3
size = max(len(minbalayalea),len(minbalayconst))

errbalayconst = []
errbalayconst = np.zeros(size)
for i in range(0,len(minbalayconst)):
for i in range(0,len(minbalayalea)):
    errbalayalea[i] = abs(minbalayalea[i]-minimum_reel)
    

x = [k for k in range(0,size)]
plt.plot(x,errbalayalea, color = 'blue')

#-------------------------------- Question 4  ---------------------------------

print("Question 4 ")
print("Par balayage constant, on obtient minimum = ")
print(balayage_constant(func, 0, 3,10000))
print()
#-------------------------------- Question 5  ---------------------------------

#Sur papier

#-------------------------------- Question 6  ---------------------------------

def gradient_1D(func,dfunc,a,b,u,n):
    x0 = b
    xn = x0
    for i in range(1,n):
        xn = xn + u * dfunc(xn)
    minimum = func(xn) 
    return minimum

print("Question 6 ")
print("Par gradient 1D, on obtient minimum = ")
print(gradient_1D(func, dfunc, 0, 3, -0.0001, 10000))
print()


#-------------------------------- Question 7  ---------------------------------

#Sur papier

#------------------------------------------------------------------------------
#         C.   Optimisation d'une fonction de deux variables réelles
#------------------------------------------------------------------------------


#-------------------------------- Question 1  ---------------------------------

def g_ab(x,y,a,b):
    return (x**2)/a + (y**2)/b

def h(x,y):
    return math.cos(x) * math.sin(y)
    
def graph_3D(func,a,b):
#Graph de la fonction
    x = [k for k in np.arange(a,b,0.0001)]
    result = []
    for i in np.arange(a,b,0.001):
        result.append(func(i))
    plt.plot(x,result)

# ax = plt.axes(projection='3d')

# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

#-------------------------------- Question 2  ---------------------------------



#-------------------------------- Question 3  ---------------------------------



#-------------------------------- Question 4  ---------------------------------



#-------------------------------- Question 5  ---------------------------------

def gradpc(eps, m, u, x0, y0, df1, df2):
    print("a")

#-------------------------------- Question 6  ---------------------------------



#-------------------------------- Question 7  ---------------------------------



#-------------------------------- Question 8  ---------------------------------

def gradamax(eps, m, u, x0, y0, f, df1, df2):
    print("a")


#-------------------------------- Question 9  ---------------------------------

def gradamin(eps, m, u, x0, y0, f, df1, df2):
    print("a")

#-------------------------------- Question 10  --------------------------------

#------------------------------------------------------------------------------
#         D.   Calcul de l'inverse d'une matrice
#------------------------------------------------------------------------------

#-------------------------------- Question 1  ---------------------------------



#-------------------------------- Question 2  ---------------------------------



#-------------------------------- Question 3  ---------------------------------



#------------------------------------------------------------------------------
#         E.   Application à des problèmes de transfert de chaleur
#------------------------------------------------------------------------------

#-------------------------------- Question 1  ---------------------------------



#-------------------------------- Question 2  ---------------------------------




# %%
