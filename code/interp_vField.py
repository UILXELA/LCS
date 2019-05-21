# Alex Liu
# 11/13/18

import pylab as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd

fig, ax = plt.subplots(1,1,figsize=(10,5))

# constants
p = np.pi
A = 0.1
epsilon = 0.25
w = p/5
delta = 0.0001
dt = 0.1
partition = 20
col = ['r','y','b','g','k','c','m','r','y','b',
       'g','k','c','m','r','y','b','g','k','c','m',
       'r','y','b','g','k','c','m']

def readcsv(t):
    vx = pd.read_csv("./csv_Data/X_Velocity %.2f.csv"%t,header=None).values
    vy = pd.read_csv("./csv_Data/Y_Velocity %.2f.csv"%t,header=None).values
    return vx,vy
# wave function that defines the characteristics of 
# double gyre
def phi(x,y,t):
    temp = A*np.sin(p*f(x,t))*np.sin(p*y)
    return temp

def f(x,t):
    temp = epsilon*np.sin(w*t)*x**2+(1-2*epsilon*np.sin(w*t))*x
    return temp

def velocity(x,y,t):
    vx = (phi(x,y+delta,t)-phi(x,y-delta,t))/(2*delta)
    vy = (phi(x-delta,y,t)-phi(x+delta,y,t))/(2*delta)
    return -1*vx,-1*vy
	
def interp(x,y,t):
    vx,vy=readcsv(t)
    top = np.ceil(y*partition).astype(int)
    #print((y*partition).astype(int))
    #print(top)
    #top = top.astype(int)
    bot = top-1
    right = np.ceil(x*partition).astype(int)
    #right = right.astype(int)
    left = right-1
    vxi = (vx[top,right]*(x*partition-left)+ vx[top,left]*(right-x*partition))*(y*partition-bot) + (vx[bot,right]*(x*partition-left) + vx[bot,left]*(right-x*partition))*(top-y*partition)  
    vyi = (vy[top,right]*(x*partition-left)+ vy[top,left]*(right-x*partition))*(y*partition-bot) + (vy[bot,right]*(x*partition-left) + vy[bot,left]*(right-x*partition))*(top-y*partition)    
    return -1*vxi,-1*vyi
def update(r,t):
    x = r[0]
    y = r[1]
    vx,vy = interp(x,y,t)
    return np.array([-1*vx,-1*vy],float)
# make a 2D mesh grid of size 40*20
X,Y = plt.meshgrid(np.arange(0,2,1/partition),np.arange(0,1,1/partition))
Vx,Vy = velocity(X,Y,0.1)
Vx2,Vy2 = interp(X,Y,0.1)

# vector arrows
Q2 = ax.quiver(X,Y,Vx2,Vy2,scale=10,color='r')
Q = ax.quiver(X,Y,Vx,Vy,scale=10)

plt.show()

