#Alex Liu

import numpy as np
import numpy.linalg as LA
import time as time
import math
import pandas as pd
import sys

# constants
row = 40
col = 60
p = np.pi
partition = 20
A = 0.1
epsilon = 0.25
w = p/5
Delta = 0.0001
delta = 0.01
dt = 0.1
T = 20
L = 400
H = 200
tau_lim = 10
data_size=2*(T+tau_lim)/dt+1
data=np.empty([int(data_size),2,row,col],dtype=np.float64)

start_time = time.time()

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
    vx = (phi(x,y+Delta,t)-phi(x,y-Delta,t))/(2*Delta)
    vy = (phi(x-Delta,y,t)-phi(x+Delta,y,t))/(2*Delta)
    return -1*vx,-1*vy

def csvtoRAM(T):
    print("Loading data onto RAM\n\n")
    i = 0
    for t in np.arange(0,T+tau_lim+0.5*dt,0.5*dt):
        data[i,0], data[i,1]=readcsv(t)
        i+=1
    if(i!=data.shape[0]):
        print("size error,i=", i, " expected", data.shape[0])
    else:
        print("Loading completed\n\n")

def interp(x,y,t):
    ind = int(2*t/dt)
    vx =data[ind,0]
    vy =data[ind,1]
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

def update(r,t,op):
    x = r[:,0]
    y = r[:,1]
    if(op=='1'):
        vx,vy = interp(x,y,t)
    else:
        vx,vy = velocity(x,y,t)
    return np.column_stack((vx,vy))

def mapping(op):  
    shift = 1.1*delta
    # y-coordinate of the grid
    h = np.linspace(shift,1,H)
    tau = np.linspace(0,tau_lim,101)
#    print(tau)
    # getting 100 mapping data files
    for n in range(0,100):
        if(op=="1"):
            output = open('./mapping_files/mapping%d.txt'%n,'ab')
        else:
            output = open('./mapping_files_model/mapping%d.txt'%n,'ab')
        local_start = time.time()
        for i in h:
	
	    # initialize grid horizontally
            x = np.linspace(shift,2,L)
            y = np.linspace(i,i,L)
            r = np.column_stack((x,y))
            print("\r",i,end='')
		  # perform RK4 to get position of particle 20s later
#            for t in np.arange(tau[n],T-tau[-1]+tau[n],dt):
            for t in np.arange(0+tau[n],T+tau[n],dt):
#                pass
#                print("t:",t)
                k1 = dt*update(r,t,op)
                k2 = dt*update(r+0.5*k1,t+0.5*dt,op)
                k3 = dt*update(r+0.5*k2,t+0.5*dt,op)
                k4 = dt*update(r+k3,t+dt,op)
                r += (k1+2*k2+2*k3+k4)/6
		
		  # append data to the file	
            np.savetxt(output,r)       
        output.close()
        print ("\n")
        print (n,". ",time.time()-local_start)
        print ("\n")
    print(time.time()-start_time)

op=input("Please type in 0 to generate mapping files for model, 1 for data\n")
if(op=='1'):
    csvtoRAM(T)
mapping(op)




