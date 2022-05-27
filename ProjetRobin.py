# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:06:10 2022

@author: Robin
"""

import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
"xB. Optimisation d'une fonction d'une variable reelle"


def f(x):
    return x**3-3*x**2+2*x+5


def Meth_Bal_Const(f, a, b, N):
    dx = (b-a)/N
    mini = f(a)
    for i in range(0, N+1):
        if mini > f(a+i*dx):
            mini = f(a+i*dx)
    # print(mini)
    return mini


def Meth_bal_alea(f, a, b, N):
    x = []

    for i in range(0, N+1):
        x.append(f(random.random()*(b-a)+a))
    mini = x[0]
    for i in range(1, N+1):
        if mini > x[i]:
            mini = x[i]
    # print(mini)
    return mini


def convergence(N):
    vrai = 1+1/math.sqrt(3)
    x = []
    y_const = []
    y_alea = []

    for i in range(1, N):
        x.append(i)
        y_const.append(abs(f(vrai)-Meth_Bal_Const(f, 0, 3, i)))
        y_alea.append(abs(f(vrai)-Meth_bal_alea(f, 0, 3, i)))
    plt.plot(x, y_const, label="erreur mÃ©thode par balayage Ã  pas constant")
    plt.plot(x, y_alea, label="erreur mÃ©thode par balayage alÃ©atoire")
    plt.legend()
    plt.show()


def Meth_Bal_Const_Max(f, a, b, N):
    dx = (b-a)/N
    maxi = f(a)
    for i in range(0, N):
        if maxi < f(a+i*dx):
            maxi = f(a+i*dx)
    print(maxi)
    return maxi


"Question 5"
"si la fonction est croissante, f' est positif, on a  x<y --> f(x)<f(y)"
"Ainsi il faut que xn+1<xn pour se rapprocher du minimum.Ce qui explique u<0 pour que uf'<0"
" On est prudent donc on choisit un petit pas u "


def minidari(f, a, b, N):
    xn = random.random()*(b-a)+a
    # x=sp.Symbol('x')
    u = -0.000001
    for i in range(1, N):
        # xn=xn+u*sp.diff(f,x,1e-6)
        xn = xn+u*((f(xn-u)-f(xn))/u)**2
    print(xn)


"xC. Optimisation d'une fonction de deux variables reelles"


def g(x, y, a, b):
    g = x**2/a+y**2/b
    return g


def h(x, y):
    return (np.cos(x))*(np.sin(y))


"Question 1:"


def graphG():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = g(X, Y, 2, 2/7)
    fig = plt.figure(figsize=(8, 6))
    ax3d = plt.axes(projection='3d')
    surf = ax3d.plot_surface(X, Y, Z, rstride=7, cstride=7, cmap="viridis")
    fig.colorbar(surf, ax=ax3d)
    ax3d.set_titel('Surface plot of g')
    ax3d.st_xlabel('X')
    ax3d.st_ylabel('Z')
    ax3d.st_zlabel('Z')

    plt.show


def graphH():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = h(X, Y)
    fig = plt.figure(figsize=(8, 6))
    ax3d = plt.axes(projection='3d')
    surf = ax3d.plot_surface(X, Y, Z, rstride=7, cstride=7, cmap="viridis")
    fig.colorbar(surf, ax=ax3d)
    ax3d.set_titel('Surface plot of h')
    ax3d.st_xlabel('X')
    ax3d.st_ylabel('Z')
    ax3d.st_zlabel('Z')

    plt.show


"Question 2 :"


def courbe_niv_g():
    feature_x = np.arange(-60, 60, 0.5)
    feature_y = np.arange(-50, 50, 0.5)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    fig, ax = plt.subplots(1, 1)
    Z = g(X, Y, 2, 2/7)
    ax.contour(X, Y, Z)
    ax.set_title('Courbe de niveau de g')
    ax.set_xlabel('feature_x')
    ax.set_ylabel('feature_y')
    plt.show()


def courbe_niv_h():
    feature_x = np.arange(-10, 10, 0.1)
    feature_y = np.arange(-10, 10, 0.1)
    [X, Y] = np.meshgrid(feature_x, feature_y)
    fig, ax = plt.subplots(1, 1)
    Z = h(X, Y)
    ax.contour(X, Y, Z)
    ax.set_title('Courbe de niveau de h ')
    ax.set_xlabel('feature_x')
    ax.set_ylabel('feature_y')
    plt.show()


"Question 3"
"grad(g) = (x,7y)"
"grad(h) = (-sin(y)*sin(x);cos(x)*cos(y))"


def grad_gx(x, y,a,b):
    g = x*2/a
    return g
def grad_gy(x, y,a,b):
    g = 1/b*(2*y)
    return g
def grad_hx(x, y):
    h = (-np.sin(y)*np.sin(x))
    return h
def grad_hy(x, y):
    h = (np.cos(x)*np.cos(y))
    return h
def norme_grad_h(x, y,a,b):
    n = math.sqrt((grad_hx(x, y,a,b))**2+grad_hy(x, y,a,b)**2)
    return n
def norme_grad_g(x, y,a,b):
    n = math.sqrt((grad_gx(x, y,a,b))**2+grad_gy(x, y,a,b)**2)
    return n


"Question 5 et 6:"


def gradpc(eps, m, u, x0, y0, df1, df2):
    cx = []  # coordonnÃ©e xn
    cy = []  # coorodnÃ©e de yn
    lx = []
    lx.append(0)
    cx.append(x0)
    cy.append(y0)
    for i in range(0, m-1):
        cx.append(cx[(i)]+u*df1(cx[(i)], cy[(i)]))
        cy.append(cy[(i)]+u*df2(cx[(i-1)], cy[(i)]))
        lx.append(i)
        if math.sqrt(df1(cx[(i-1)], cy[(i-1)])**2+df2(cx[(i-1)], cy[(i-1)])**2) <= eps:
            i = m-1
    plt.plot(cx, cy, label='gradiant de la fonction')
    plt.show
    return


def gradpc_h(eps, m, u, x0, y0, df1, df2,a,b):  # m=100,u=0,2
    cx = []  # coordonnÃ©e xn
    cy = []  # coorodnÃ©e de yn
    lx = []
    lx.append(0)
    cx.append(x0)
    cy.append(y0)
    for i in range(0, m-1):
        cx.append(cx[(i)]+u*df1(cx[(i)], cy[(i)]))
        cy.append(cy[(i)]+u*df2(cx[(i-1)], cy[(i)]))
        lx.append(i)
        if math.sqrt(df1(cx[(i-1)], cy[(i-1)])**2+df2(cx[(i-1)], cy[(i-1)])**2) <= eps:
            i = m-1
    courbeniv_x = np.arange(-5, 5, 0.1)
    courbeniv_y = np.arange(-5, 5, 0.1)
    [X, Y] = np.meshgrid(courbeniv_x, courbeniv_y)
    fig, ax = plt.subplots(1, 1)
    Z = h(X, Y)
    ax.contour(X, Y, Z)
    plt.plot(cx, cy, label='gradiant de la fonction', color='r')
    ax.set_title('Courbe de niveau de h avec le gradient en rouge ')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def gradpc_g(eps, m, u, x0, y0, df1, df2,a,b):  # m=6,u=0,2
    cx = []  # coordonnÃ©e xn
    cy = []  # coorodnÃ©e de yn
    lx = []
    lx.append(0)
    cx.append(x0)
    cy.append(y0)
    for i in range(0, m-1):
        cx.append(cx[(i)]+u*df1(cx[(i)], cy[(i)],a,b))
        cy.append(cy[(i)]+u*df2(cx[(i-1)], cy[(i)],a,b))
        lx.append(i)
        if math.sqrt(df1(cx[(i-1)], cy[(i-1)],a,b)**2+df2(cx[(i-1)], cy[(i-1)],a,b)**2) <= eps:
            i = m-1
    courbeniv_x = np.arange(-60, 50, 1)
    courbeniv_y = np.arange(-100, 250, 1)
    [X, Y] = np.meshgrid(courbeniv_x, courbeniv_y)
    fig, ax = plt.subplots(1, 1)
    Z = g(X, Y, a, b)
    ax.contour(X, Y, Z)
    plt.plot(cx, cy, color='r')
    ax.set_title('Courbe de niveau de g avec le gradient en rouge ')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return cx[m-1],cy[m-1]


"Question 7"


def erreur_r(eps, m, u, x0, y0, df1, df2,a,b):
    cx = []  # coordonnÃ©e xn
    cy = []  # coorodnÃ©e de yn
    lx = []
    lg=[]
    cx.append(x0)
    cy.append(y0)
    for i in range(0, m-1):
        cx.append(cx[(i)]+u*df1(cx[(i)], cy[(i)],a,b))
        cy.append(cy[(i)]+u*df2(cx[(i)], cy[(i)],a,b))
        lg.append(g(cx[i],cy[i],a,b))
        lx.append(i)
        if math.sqrt(df1(cx[(i-1)], cy[(i-1)],a,b)**2+df2(cx[(i-1)], cy[(i-1)],a,b)**2) <= eps:
            i = m-1
        print('valeur de lgi :' ,lg[i-1],i-1)
    vvx,vvy=0,0
    #gradpc_g(eps, m, u, x0, y0, df1, df2, a, b)
    gvrai = g(vvx,vvy,a,b)
    e = []
    for i in range(0, m-1):
            e.append(abs(gvrai-lg[i]))
    plt.plot(lx,e,'r')
    plt.title('courbe erreur Ã  pas constant')
    plt.show

"Il faut que u<-0,10 pour que l'erreur relative soit dÃ©croissante, je ne suis pas sur de mon erreur relative"
"Question 8 :"

def gradamax_h(eps,m,u,x0,y0,f,df1,df2):#m=120, u = 0.1,x0=1,y0=1
    k=1
    i=0
    Vx=[]
    Vy=[]
    Vx.append(x0)
    Vy.append(y0)
    for i in range(0,m-1): 
        g1=df1(Vx[i],Vy[i])
        g2=df2(Vx[i],Vy[i])
        F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2)
        F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2)
        while F1<F2:
            k=k+0.1
            F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2)
            F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2)
        print(Vx[i],Vy[i])
        Vx.append(Vx[i]+k*u*df1(Vx[i],Vy[i]))
        Vy.append(Vy[i]+k*u*df2(Vx[i],Vy[i]))#est ce que il y a k+1 ? 
        if math.sqrt(df1(Vx[(i-1)], Vy[(i-1)])**2+df2(Vx[(i-1)], Vy[(i-1)])**2) <= eps:
            i = m-1
    courbeniv_x = np.arange(-5, 5, 0.1)
    courbeniv_y = np.arange(0.5, 2, 0.1)
    [X, Y] = np.meshgrid(courbeniv_x, courbeniv_y)
    fig, ax = plt.subplots(1, 1)
    Z = h(X, Y)
    ax.contour(X, Y, Z)
    plt.plot(Vx, Vy,'r.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Courbe de niveau de h avec le gradient en point rouge')
    plt.plot(Vx,Vy,'r.')
    plt.show
    

"Question 8 et 9 "

def gradamax_g(eps,m,u,x0,y0,f,df1,df2,a,b):#m=6 u=0.2 , x0=0.9, y0 = 0.9 
    k=1
    i=0
    Vx=[]
    Vy=[]
    c=0
    Vx.append(x0)
    Vy.append(y0)
    for i in range(0,m-1): 
        c=i
        g1=df1(Vx[i],Vy[i],a,b)
        g2=df2(Vx[i],Vy[i],a,b)
        F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2,a,b)
        #print(F1)
        F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2,a,b)
        #print(F2)
        #print(1)
        while F1<F2:
            k=k+0.1
            F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2,a,b)
            F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2,a,b)
            print(1)
        #print(Vx[i],Vy[i])
        Vx.append(Vx[i]+k*u*df1(Vx[i],Vy[i],a,b))
        Vy.append(Vy[i]+k*u*df2(Vx[i],Vy[i],a,b))
        if math.sqrt(df1(Vx[(i-1)], Vy[(i-1)],a,b)**2+df2(Vx[(i-1)], Vy[(i-1)],a,b)**2) <= eps:
            i = m-1
    courbeniv_x = np.arange(-60, 50, 0.1)
    courbeniv_y = np.arange(-100, 100, 0.1)
    [X, Y] = np.meshgrid(courbeniv_x, courbeniv_y)
    fig, ax = plt.subplots(1, 1)
    Z = g(X, Y,a,b)
    ax.contour(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Courbe de niveau de g avec le gradient en point rouge')
    plt.plot(Vx,Vy,'r.')
    plt.show
    return c
    
def gradamini_h(eps,m,u,x0,y0,f,df1,df2):#m=120, u = -0.2,x0=2,y0=2
    k=1
    i=0
    Vx=[]
    Vy=[]
    Vx.append(x0)
    Vy.append(y0)
    for i in range(0,m-1): 
        g1=df1(Vx[i],Vy[i])
        g2=df2(Vx[i],Vy[i])
        F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2)
        F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2)
        while F1>F2:
            k=k+0.1
            F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2)
            F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2)
        print(Vx[i],Vy[i])
        Vx.append(Vx[i]+k*u*df1(Vx[i],Vy[i]))
        Vy.append(Vy[i]+k*u*df2(Vx[i],Vy[i]))
        if math.sqrt(df1(Vx[(i-1)], Vy[(i-1)])**2+df2(Vx[(i-1)], Vy[(i-1)])**2) <= eps:
            i = m-1
    courbeniv_x = np.arange(0, 5,0.1)
    courbeniv_y = np.arange(0, 3, 0.1)
    [X, Y] = np.meshgrid(courbeniv_x, courbeniv_y)
    fig, ax = plt.subplots(1, 1)
    Z = h(X, Y)
    ax.contour(X, Y, Z)
    plt.plot(Vx, Vy,'r.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Courbe de niveau de h avec le gradient en point rouge')
    plt.plot(Vx,Vy,'r.')
    plt.show
    
  

def gradamini_g(eps,m,u,x0,y0,f,df1,df2,a,b):#m=6 u=-0.2 , x0=50, y0 = 30 
    k=1
    i=0
    Vx=[]
    Vy=[]
    Vx.append(x0)
    Vy.append(y0)
    for i in range(0,m-1): 
        g1=df1(Vx[i],Vy[i],a,b)
        g2=df2(Vx[i],Vy[i],a,b)
        F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2,a,b)
        #print(F1)
        F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2,a,b)
        #print(F2)
        #print(1)
        while F1>F2:
            k=k+0.1
            F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2,a,b)
            F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2,a,b)
            #print(1)
        print(Vx[i],Vy[i])
        Vx.append(Vx[i]+k*u*df1(Vx[i],Vy[i],a,b))
        Vy.append(Vy[i]+k*u*df2(Vx[i],Vy[i],a,b))
        if math.sqrt(df1(Vx[(i-1)], Vy[(i-1)],a,b)**2+df2(Vx[(i-1)], Vy[(i-1)],a,b)**2) <= eps:
            i = m-1
    courbeniv_x = np.arange(-60, 50, 0.1)
    courbeniv_y = np.arange(-100, 100, 0.1)
    [X, Y] = np.meshgrid(courbeniv_x, courbeniv_y)
    fig, ax = plt.subplots(1, 1)
    Z = g(X, Y,a,b)
    ax.contour(X, Y, Z)
    plt.plot(Vx, Vy,'r.')
    ax.set_title('Courbe de niveau de g avec le gradient en point rouge')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show
    
print(gradamini_g(0.00001,60,-0.2,50,30,g,grad_gx,grad_gy,2,2/7))

    
def gradamini_g(eps,u,x0,y0,f,df1,df2,a,b):#m=6 u=-0.2 , x0=50, y0 = 30 

        k=0.1
        i=0
        drapeau = 0
        Vx=[]
        Vy=[]
        Vx.append(x0)
        Vy.append(y0)
        c=0
        while drapeau == 0:
            c=c+1
            g1=df1(Vx[i],Vy[i],a,b)
            g2=df2(Vx[i],Vy[i],a,b)
            F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2,a,b)
            #print(F1)
            F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2,a,b)
           # print(F2)
            while F1>F2:
                k=k+0.1
                F1=f(Vx[i]+k*u*g1,Vy[i]+k*u*g2,a,b)
                F2=f(Vx[i]+(k+1)*u*g1,Vy[i]+(k+1)*u*g2,a,b)
                #print(1)
            print(Vx[i],Vy[i])
            #on rÃ©initialise k
            Vx.append(Vx[i]+k*u*df1(Vx[i],Vy[i],a,b))
            Vy.append(Vy[i]+k*u*df2(Vx[i],Vy[i],a,b))
            #on rÃ©initialise k
            k=0.1
            if math.sqrt(df1(Vx[(i-1)], Vy[(i-1)],a,b)**2+df2(Vx[(i-1)], Vy[(i-1)],a,b)**2) <= eps:
                drapeau = 1
           # if math.sqrt((Vx[i])**2+Vy[i]**2)<=eps:
            #    drapeau=1
            i = i+1
            if i > 120:
                drapeau= 1
                c=200
        #print ('retour de c: ',c , 'valeur de u :',u)
        return c
    

def gradpc_g2(eps,u,x0,y0,f,df1,df2,a,b):
    cx = []  # coordonnÃ©e xn
    cy = []  # coordonnÃ©e de yn
    i=0
    drapeau=0
    cx.append(x0)
    cy.append(y0)
    while drapeau ==0:
        cx.append(cx[(i)]+u*df1(cx[(i)], cy[(i)],a,b))
        cy.append(cy[(i)]+u*df2(cx[(i-1)], cy[(i)],a,b))
        i=i+1
        if math.sqrt(df1(cx[(i-1)], cy[(i-1)],a,b)**2+df2(cx[(i-1)], cy[(i-1)],a,b)**2) <= eps:
            drapeau = 1
        if i>=2000:
            drapeau = 1
            i=2000
    return i
    
" Question 10 "
def erreur_rpa(eps,m,x0,y0,f,df1,df2):
    a=1
    b=20
    c=[]
    U=[]
    u=-0.999
    while u<=0.001:
        c.append(gradamini_g2(eps,u,x0,y0,f,df1,df2,a,b))
        U.append(u)
        u=u+0.001
    plt.plot(U,c)
    plt.show

def erreur_comp(eps,x0,y0,f,df1,df2):
    a=1
    b=20
    cpa=[]
    cpc=[]
    U=[]
    u=-0.999
    while u<=0.001:
        cpa.append(gradamini_g2(eps,u,x0,y0,f,df1,df2,a,b))
        cpc.append(gradpc_g2(eps,u, x0, y0,f, df1, df2, a, b))
        U.append(u)
        u=u+0.001
    plt.plot(U,cpa,'r')
    plt.plot(U,cpc,'g')
    plt.show
    
    
    
"xD : Calcul de l'inverse d'une matrice : "

def G(Y,A,B):
    G=A.dot(Y)
    #print ('1' ,G)
    G1=G-B
    #print(G1,'2')
    G2=G1*2
    print('Gradiant : ',G2)
    return G2
##appliquÃ© la definition de la norme de Frobenuis 
##norme d'une matrice est Ã©gale Ã  la trace du produit de M* avec M et  M* la transconjuguÃ©e
def norme(Y):
    n=(Y.conj().transpose())
    #print(n)
    n=n.dot(Y)
    #print(n)
    if n.ndim > 1: 
        n=n.trace()**(1/2)
        print(1)
    else : 
        n=n
        print(2)
    return n
#norme pour un vecteur
def norme2(Y,n):
    N=0
    for i in range (0,n):
        #print(Y[i][0] ,'normeY')
        N=N+(Y[i][0])**2
    N=np.sqrt(N)
    #print(N)
    return N 
    
def Ag(a,b):
    return np.array([[1/a,0],[0,1/b]])
def bg():
    return np.array([[0],[0]])
def yg(x,y):
    Y= np.array([[x],[y]])
    print(Y)
    return Y
def test(x,y):
    A=Ag(x,y)
    M=A.dot(yg(x,y))
    print(M)
    
def Mat(Y,X,i):#transforme deux listes en un vecteur au coordonnÃ©e des positions i 
    Y=np.array([[X[i]],[Y[i]]])
    #print(Y)
    return Y
"Question 2"

def algopo(eps,m,Y,A,B):
    pk=[]
    i=0
    drapeau=0
    while drapeau==0:
        #print(i)
        g=G(Y,A,B)
        if  (g!=0).all() == True :
            print(1)
            div1=(g.transpose()).dot(A)
            div=div1.dot(g)
            div=2*div
            pk.append(norme(g)**2/div[0][0])
            print(pk,'2')
        else:
            pk.append(0)
        Y2=Y-pk[i]*g
        M=Y2-Y
        print(M)
        if norme(M)<= eps:
            j=i
            drapeau=1
            print('tolÃ©rance atteinte Ã  la i eme itÃ©rations :',j)
        if i>= m:
            drapeau=1
        else:
            i=i+1
            Y=Y2
    print(Y)           

"On rentre Y un vecteur coordonnÃ©e de depart, A ,B deifnit pour G le gradiant et n la dimension"
"m le nombre d'itÃ©ration max"
def algopoprime(eps,n,m,Y,A,B):
        pk=[]
        i=0
        drapeau=0
        while drapeau==0:
            print(i)
            g=G(Y,A,B)
            gt=g.transpose()
            #print('Gradiant t', gt )
            if  (g!=0).any() == True :
                div1=gt.dot(A)
                div=div1.dot(g)
                div=2*div
                #print(div[0][0],'div0')
                
                pk.append(norme2(g,n)**2/div[0][0])
                #print(pk[i],'pk')
            else:
                pk.append(0)
            Y2=Y-pk[i]*g
            M=Y2-Y
            #print(M,'M')
            if norme2(M,n)<= eps:
                j=i
                drapeau=1
                print('tolÃ©rance atteinte Ã  la i eme itÃ©rations :',j)
            if i>= m:
                drapeau=1
            else:
                i=i+1
                Y=Y2
        print(Y,'Y')
        return Y 
"Question 3 :"
#on adapte l'algo Ã  n=2
def algopo2(eps,m,x0,y0,a,b,B):
    a=1
    b=20
    A=Ag(a,b)
    x=[]
    x.append(x0)
    y=[]
    y.append(y0)
    Y=Mat(y, x, 0) 
    #print('Y = ',Y)
    pk=[]
    i=0
    drapeau =0
    while drapeau==0:
        #print(i)
        g=G(Y,A,B)
        #print('g : ',g)
        if  (g!=0).all() == True :
            div1=(g.transpose()).dot(A)
            #print('div1 :',div1)
            div=div1.dot(g)
            div=2*div
           #print('div : ',div[0][0])
            pk.append(norme(g)**2/div[0][0])
        else:
            pk.append(0)
        #print('pk au rang i :',pk[i], i )
        gx=g[0,:][0]
        gy=g[1,:][0]
        x.append(x[i]-pk[i]*gx)
        y.append(y[i]-pk[i]*gy)
        print(x[i],y[i])
        Y=Mat(y,x,i+1)
        #print('Y apres la ieme iterations : ',Y,i)
        M=np.array([[x[i+1]-x[i]],[y[i+1]-y[i]]])
        #print('norme :' ,norme(M))
        if norme(M)<= eps:
            j=i
            drapeau=1
            print('tolÃ©rance atteinte Ã  la i eme itÃ©rations :',j)
        if i>= m:
            drapeau=1
        else:
            i=i+1
    plt.plot(x,y,'.r')
    plt.show
    
"Partie E"

"Question1"
#marche rempli la matrice A du systeme 
def AE1(n,l):
    deltax = l/(n+1)
    v1 =[]
    v2 = []
    v3 = []
    #Matrice Ã  inverser
    v1.append(-2)
    for i in range(1,n-1):
        v1.append(-2)
    v1.append(-2)
    for i in range(0,n-2):
        v2.append(1)
    v2.append(1)
    v3.append(1)
    for i in range(0,n-2):
        v3.append(1)
    A = np.diag(v1)+np.diag(v2,-1)+np.diag(v3,1)
    return A

"Question 2"

def AE2(n,l,d):
    dx = l/(n+1)
    v1 =[]
    v2 = []
    v3 = []
    #Matrice Ã  inverser
    v1.append(2)
    for i in range(1,n-1):
        v1.append(2)
    v1.append(2)
    for i in range(0,n-2):
        v2.append(-1)
    v2.append(-1)
    v3.append(-1)
    for i in range(0,n-2):
        v3.append(-1)
    A = np.diag(v1)+np.diag(v2,-1)+np.diag(v3,1)
    A=A*(1/(dx**2))
    v4=[]
    for i in range(0,n):
        v4.append((1/d)**2)
    A2=np.diag(v4)
    A=A+A2
    return A
def BE2(n,Ta,a,b,d,l):
    v=[]
    O=[]
    dx=l/(n+1)
    D=(1/d)**2
    v.append(D*Ta+(a/(dx**2)))
    O.append(0)
    for i in range(1,n-1):
        v.append(D*Ta)
        O.append(0)
    O.append(0)
    v.append(D*Ta+(b/(dx**2)))
    B=np.array([v])
    B=np.transpose(B)
    #print(B)
    return B
"meme condition que pour algopoprime mais l en plus la longueur de la barre "
def graphe2(eps, n, m, Y, A, B,l):
    Y=algopoprime(eps, n, m, Y, A, B)
    dx=l/(n+1)
    x=[]
    for i in range(0,n):
        x.append(i*dx)
    plt.plot(x,Y)
    plt.show
    