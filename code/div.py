# Alex Liu
# 11/13/18

import pylab as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd
import matplotlib.cm as cm
from matplotlib import colors


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
T=200

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
    vx = np.array((phi(x,y+delta,t)-phi(x,y-delta,t))/(2*delta))
    vy = np.array((phi(x-delta,y,t)-phi(x+delta,y,t))/(2*delta))
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
    return np.array(-1*vxi),np.array(-1*vyi)
def update(r,t):
    x = r[0]
    y = r[1]
    vx,vy = interp(x,y,t)
    return np.array([-1*vx,-1*vy],float)
# make a 2D mesh grid of size 40*20
X,Y = np.array(plt.meshgrid(np.arange(0,2,1/partition),np.arange(0,1,1/partition)))

t=float(input("Please type in the time point you want to see the divergence at. ([0,%d])\n"%T))
Vx,Vy = velocity(X,Y,t)
Vx2,Vy2 = interp(X,Y,t)
divInterp=np.array(np.gradient(Vx2)[1]+np.gradient(Vy2)[0])
divModel=np.array(np.gradient(Vx)[1]+np.gradient(Vy)[0])

f = open("div.txt", "w")
print("Time: %.1f\n"%t, file=f)
print("Model:\n", file=f)
print(divModel, file=f)
print("\n*************************************************************************************\n", file=f)
print("Data:\n", file=f)
print(divInterp, file=f)
fig, ax = plt.subplots(2, 1)
pcm = ax[0].pcolormesh(X, Y, divModel, cmap=cm.jet)
ax[0].set_title("Divergence of Model at t=%.1f"%t)
fig.colorbar(pcm, ax=ax[0], extend='max')

pcm = ax[1].pcolormesh(X, Y, divInterp, cmap=cm.jet)
ax[1].set_title("Divergence of Data at t=%.1f"%t)
fig.colorbar(pcm, ax=ax[1], extend='max')
plt.show()

