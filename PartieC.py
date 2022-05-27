#%%

from turtle import color
from matplotlib import colors
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import random

#------------------------------------------------------------------------------
#         C.   Optimisation d'une fonction de deux variables réelles
#------------------------------------------------------------------------------

print("C.   Optimisation d'une fonction de deux variables réelles")

#-------------------------------- Question 1  ---------------------------------
print("Question 1")

def g_ab(x,y,a,b):
    return (x**2)/a + (y**2)/b

def g_227(x,y):
    return (x**2)/2 + (y**2)/(2/7)

def h(x,y):
    return np.cos(x) * np.sin(y)
    
#Graph de g_ab
def plot_3D_gab(a,b):
    plt.ion()
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = g_ab(X,Y,a,b)
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='coolwarm', edgecolor='none')
    ax.set_title("g(x,y)_2,2/7", fontsize = 13)
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)

#Graph d'une fonction (ici h)
def plot_3D(func):
    x = np.linspace(-10, 10,200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X,Y)
    fig2 = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='coolwarm', edgecolor='none')
    ax.set_title("h(x,y)", fontsize = 13)
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)
    
plot_3D_gab(2,2/7)
plot_3D(h)
print()

#-------------------------------- Question 2  ---------------------------------

print("Question 2")

def plot_3D_LN_gab(a,b):
    plt.ion()
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = g_ab(X,Y,a,b)
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='jet', edgecolor='none', alpha=0)
    ax.set_title("contour g(x,y)_2,2/7", fontsize = 13)
    ax.contour(X, Y, Z, 20, colors="k", linestyles="solid")
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)
    
def plot_3D_LN(func):
    x = np.linspace(-10, 10,200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X,Y)
    fig2 = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap='coolwarm', edgecolor='none', alpha=0)
    ax.set_title("contour h(x,y)", fontsize = 13)
    ax.contour(X, Y, Z, 20, colors="k", linestyles="solid")
    ax.set_xlabel('x', fontsize = 11)
    ax.set_ylabel('y', fontsize = 11)
    ax.set_zlabel('Z', fontsize = 10)    

def plot_2D_LN(func):
    f = np.vectorize(func)
    X = np.arange(-10, 10, 0.01)
    Y = np.arange(-10, 10, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    fig3 = plt.figure(figsize = (10,10))
    plt.axis('equal')
    plt.contour(X, Y, Z, 15)
    plt.show()

def plot_2D_LN_gab(a,b):
    X = np.arange(-10, 10, 0.01)
    Y = np.arange(-10, 10, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = g_ab(X, Y, a, b)
    fig4 = plt.figure(figsize = (10,10))
    plt.axis('equal')
    plt.contour(X, Y, Z, 15)
    plt.show()

plot_3D_LN_gab(2,2/7)
plot_3D_LN(h)
plot_2D_LN_gab(2,2/7)
plot_2D_LN(h)
#%%
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

def dg_227_dy(x,y):
    return 7 * y 

def dg_ab_dy(a,b,x,y):
    return(2 * y) / b 

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
    #[ 7 ,  1.5],
    #[ 10 , 10 ],
    #[-10 , -10],
    #[-54 , 20 ],
    [42  , 58 ],
    [100 , -26]
]

def q6():
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

print("Question 5")

X = []
Y = []

def gradpc(eps, m, u, x0, y0, df1, df2):
    X = []
    Y = []
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    while (norme_gradient(grad)>eps) and (nb_iteration < m) :
        X.append(point[0])
        Y.append(point[1])
        point = point + u * grad
        grad[0] = df1(point[0],point[1])
        grad[1] = df2(point[0],point[1])
        nb_iteration += 1
    plt.plot(X,Y)
    return point


g = gradpc(0.0001,100,-0.1,0,0,dg_227_dx,dg_227_dy)
print(g)
print(g_227(g[0],g[1]))
print()

#-------------------------------- Question 6  ---------------------------------

print("Question 6")

print("Pour h(x,y) avec x0 = 0 et y0 = 0 :")
p = gradpc(0.0001,1000,-0.1,0,0,dh_dx,dh_dy)
print(p)
print("h(x,y) = ")
print( h(p[0],p[1]) )

print("Pour g227(x,y) avec x0 = 7 et y0 = 1.5 :")
print(gradpc(0.0001,1000,-0.01,7,1.5,dg_227_dx,dg_227_dy))
print()
#Le minimum global de g_22/7 est obtenu pour le couple (0,0)

#-------------------------------- Question 7  ---------------------------------


#-------------------------------- Question 8  ---------------------------------

print("Question 8")
def F1(x,y,k,u,func, grad):
    x1 = x + k * u * grad[0]
    y1 = y + k * u * grad[1]
    return func(x1,y1)

def F2(x,y,k,u,func, grad):
    x1 = x + (k + 1) * u * grad[0]
    y1 = y + (k + 1) * u * grad[1]
    return func(x1,y1)

def gradamax2(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>=eps) and (nb_iteration <= m) :
        while(f1<f2):
            k+=0.1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
        for i in range(len(grad)):
                point[i] = point[i] + u * grad[i]
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        nb_iteration += 1
    return point

def gradamax(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k=0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        k = 0
        while(f1<f2):
            k += 1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
        
        point = point + k * u * grad
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        
        nb_iteration += 1
    return point

print("Pour h(x,y) avec x0 = 0 et y0 = 0 :")
p = gradamax(0.01,100,0.01,0,0,h,dh_dx,dh_dy)
print(p)
print("h(x,y) = ")
print(h(p[0],p[1]))

#-------------------------------- Question 9  ---------------------------------

print("Question 9")

def gradamin(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k=0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    while (norme_gradient(grad)>eps) and (nb_iteration <= m) :
        k = 0
        print(point)
        while(f1>f2):
            k += 1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
        
        point = point + k * u * grad
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        
        nb_iteration += 1
    return point

def gradamin2(eps, m, u, x0, y0, f, df1, df2):
    nb_iteration = 0
    grad = np.zeros(2)
    grad[0] = df1(x0,y0) 
    grad[1] = df2(x0,y0)
    point = [x0 , y0]
    k = 0
    f1 = F1(point[0],point[1],k,u,f,grad)
    f2 = F2(point[0],point[1],k,u,f,grad)
    
    while (norme_gradient(grad)>=eps) and (nb_iteration <= m) :

        point = point + k * grad
        grad[0] = df1(point[0],point[1])  
        grad[1] = df2(point[0],point[1])
        
        while(f1>f2):
            k+=0.1
            f1 = F1(point[0],point[1],k,u,f,grad)
            f2 = F2(point[0],point[1],k,u,f,grad)
            
        nb_iteration += 1
        
    return point

print("Pour h(x,y) avec x0 = 0 et y0 = 0 :")
p = gradamin(0.01,10,-0.1,0,0,h,dh_dx,dh_dy)
print(p)
print("h(x,y) = ")
print(h(p[0],p[1]))

print("Pour g227(x,y) avec x0 = 7 et y0 = 1.5 :")
#p =gradamin(0.0001,1000,-10,7,1.5,g_227,dg_227_dx,dg_227_dy)
print(p)
print()
print("g(x,y) = ")
print(g_227(p[0],p[1]))
print()

"""def ft(x,y):
    return x**2 + y**2

def dft_dx(x,y):
    return 2*x

def dft_dy(x,y):
    return 2*y


print("Pour g227(x,y) avec x0 = 7 et y0 = 1.5 :")
p = gradamin(0.1,50,-0.1,7,1.5,ft,dft_dx,dft_dy)
print(p)"""
# %%
