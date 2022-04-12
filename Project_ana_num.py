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

print("B.   Optimisation d'une fonction d'une variable réelle")
print()

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
    errbalayconst[i] = abs(minbalayconst[i]-minimum_reel)
errbalayalea = []
errbalayalea = np.zeros(size)
for i in range(0,len(minbalayalea)):
    errbalayalea[i] = abs(minbalayalea[i]-minimum_reel)
    

x = [k for k in range(0,size)]
plt.plot(x,errbalayconst, color = 'red')
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

print("C.   Optimisation d'une fonction de deux variables réelles")

#-------------------------------- Question 1  ---------------------------------

def g_ab(x,y,a,b):
    return (x**2)/a + (y**2)/b

def g_227(x,y):
    return (x**2)/2 + (y**2)/(2/7)

def h(x,y):
    return np.cos(x) * np.sin(y)
    
#Graph de g_ab
def plot_g_ab(a,b):
    plt.ion()
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = g_ab(X,Y,a,b)
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='jet', edgecolor='none')
    ax.set_title("g(x,y)_2,2/7", fontsize = 13)
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)
    
   


#Graph de h
def plot_3D(func):
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X,Y)
    fig2 = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='jet', edgecolor='none')
    ax.set_title("h(x,y)", fontsize = 13)
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)
    
print("Question 1")
plot_g_ab(2,2/7)
plot_3D(h)
print()



#-------------------------------- Question 2  ---------------------------------

print("Question 2")
#j'ai pas trop compris 

#-------------------------------- Question 3  ---------------------------------

def grad_g_ab(a,b,x,y):
    grad = np.zeros(2)
    grad[0] = dg_ab_dx(a,b,x,y) #dg_ab/dx
    grad[1] = dg_ab_dy(a,b,x,y) #dg_ab/dy
    return grad

def dg_ab_dx(a,b,x,y):
    return (2 * x) / a

def dg_227_dx(x,y):
    return x

def dg_ab_dy(a,b,x,y):
    return(2 * y) / b

def dg_227_dy(x,y):
    return (7 * y) 

def grad_h(x,y):
    grad = np.zeros(2)
    grad[0] = dh_dx(x,y) #dh/dx
    grad[1] = dh_dy(x,y) #dh/dy
    return grad

def dh_dx(x,y):
    return - np.sin(x) * np.sin(y)

def dh_dy(x,y):
    return np.cos(x) * np.cos(y)

def affiche_gradient(tab):
    for i in tab:
        print("[ "+ str(i) + " ]")

#-------------------------------- Question 4  ---------------------------------

def norme_gradient(tab):
    return np.sqrt(tab[0]**2 + tab[1]**2)

tab_2d = [
    [  0 ,  0 ],
    [ 7 ,  1.5],
    [ 10 , 10 ],
    [-10 , -10],
    [-54 , 20 ],
    [42  , 58 ],
    [100 , -26]
]

print("Calculs du gradient et de sa norme de h(x,y) en quelques points :")
for i in tab_2d:
    grad = grad_h(i[0],i[1])
    print("x = ", str(i[0]) + "; y = "+ str(i[1]))
    print("gradient = ")
    affiche_gradient(grad)
    print("Sa norme est  : " + str(norme_gradient(grad)))
    print()
    
print("Calculs du gradient et de sa norme de g_2,2/7(x,y) en quelques points :")
for i in tab_2d:
    grad = grad_g_ab(2,2/7,i[0],i[1])
    print("x = ", str(i[0]) + "; y = "+ str(i[1]))
    print("gradient = ")
    affiche_gradient(grad)
    print("Sa norme est  : " + str(norme_gradient(grad)))
    print()
        

#-------------------------------- Question 5  ---------------------------------

def gradpc(eps, m, u, x0, y0, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        for i in range(len(grad)):
            point[i] = point[i] + u * grad[i]
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        nb_iteration += 1
    return point
        
print(gradpc(0.001,1,-0.001,-5,-5,dg_227_dx,dg_227_dy))
print()

#-------------------------------- Question 6  ---------------------------------

print("Question 6")

print("Pour h(x,y) avec x0 = 0 et y0 = 0 :")
print(gradpc(0.001,10000,-0.001,0,0,dh_dx,dh_dy))
print()

print("Pour g227(x,y) avec x0 = 7 et y0 = 1.5 :")
print(gradpc(0.001,10000,-0.001,7,1.5,dg_227_dx,dg_227_dy))
print()
#Le minimum global de g_22/7 est obtenu pour le couple (0,0)

#-------------------------------- Question 7  ---------------------------------



#-------------------------------- Question 8  ---------------------------------
#marche pas
def F1(x,y,k,u,func, grad):
    x1 = x + k * u * grad[0]
    y1 = y + k * u * grad[1]
    return func(x1,y1)

def F2(x,y,k,u,func, grad):
    x1 = x + (k + 1) * u * grad[0]
    y1 = y + (k + 1) * u * grad[1]
    return func(x1,y1)

def gradamax(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k = 0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        while(f1<f2):
            k+=0.001
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
    nb_iteration += 1
    return point


#-------------------------------- Question 9  ---------------------------------
#marche pas
def gradamin(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k = 0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        while(f1>f2):
            k+=1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
            print(f1)
            print(f2)
            print()
    nb_iteration += 1
    return point

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
