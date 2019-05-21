# Jialun Bao
# Jerry Qiu
# 11/13/15

import pylab as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd

fig, ax = plt.subplots(1,1,figsize=(10,5))
#plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin/ffmpeg'
mywriter = animation.FFMpegWriter()

# constants
p = np.pi
A = 0.1
epsilon = 0.25
w = p/5
delta = 0.0001
dt = 0.1
partition = 20
N=2
col = ['r','y','b','g','k','c','m','r','y','b',
       'g','k','c','m','r','y','b','g','k','c','m',
       'r','y','b','g','k','c','m']
#Grid
#!!!Must be the smae through all files!!!
Xmin=0
Xmax=2
Ymin=0
Ymax=1

T=200 #Simulation time length
vx_all=[]
vy_all=[]

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

def csvtoRAM(T):
    print("Loading data\n\n")
    for t in np.arange(0,T,1/partition):
        x_temp,y_temp=readcsv(t)
        vx_all.append(x_temp)
        vy_all.append(y_temp)
        t_save=t
    print("Loading completed\n\n")
    return vx_all, vy_all

def interp(x,y,t):
    vx,vy=vx_all[int(t*partition)],vy_all[int(t*partition)]
    top = np.ceil(y*partition).astype(int)
    bot = top-1
    right = np.ceil(x*partition).astype(int)
    left = right-1
    vxi = (vx[top,right]*(x*partition-left)+ vx[top,left]*(right-x*partition))*(y*partition-bot) + (vx[bot,right]*(x*partition-left) + vx[bot,left]*(right-x*partition))*(top-y*partition)  
    vyi = (vy[top,right]*(x*partition-left)+ vy[top,left]*(right-x*partition))*(y*partition-bot) + (vy[bot,right]*(x*partition-left) + vy[bot,left]*(right-x*partition))*(top-y*partition)    
    return -1*vxi,-1*vyi

def update(r,t):
    x = r[0]
    y = r[1]
    vx,vy = interp(x,y,t)
    return np.array([vx,vy],float)
def model_update(r,t):
    x = r[0]
    y = r[1]
    vx = (phi(x,y+delta,t)-phi(x,y-delta,t))/(2*delta)
    vy = (phi(x-delta,y,t)-phi(x+delta,y,t))/(2*delta)
    return np.array([-1*vx,-1*vy],float)
    
# animation for particle moving along the vector field
def animate(num,Q,Q2,Q3,X,Y,C,R,N):
    t = num/partition
    dt = 1/10
    Vx,Vy = velocity(X,Y,t)
    Vx2,Vy2 = interp(X,Y,t)
    Vx3 = R[1][0]-R[0][0]
    Vy3 = R[1][1]-R[0][1]
    Q.set_UVC(Vx,Vy)  
    Q2.set_UVC(Vx2,Vy2)
    Q3.set_UVC(Vx3,Vy3)
	# update particles' positions
    for i in range(0,N-1):
        for j in range(0,10):
            r = R[i][:]
            k1 = dt*update(r,t)
            k2 = dt*update(r+0.5*k1,t+0.5*dt)
            k3 = dt*update(r+0.5*k2,t+0.5*dt)
            k4 = dt*update(r+k3,t+dt)
            R[i][:] += (k1+2*k2+2*k3+k4)/6
    
        C[i].center = (R[i][0],R[i][1])
        i+=1
        for j in range(0,10):
            r = R[i][:]
            k1 = dt*model_update(r,t)
            k2 = dt*model_update(r+0.5*k1,t+0.5*dt)
            k3 = dt*model_update(r+0.5*k2,t+0.5*dt)
            k4 = dt*model_update(r+k3,t+dt)
            R[i][:] += (k1+2*k2+2*k3+k4)/6
    
        C[i].center = (R[i][0],R[i][1])
    return Q,Q2,Q3,C


#Load data into RAM
vx_all,vy_all=csvtoRAM(T)
print(len(vx_all))


# make a 2D mesh grid of size 40*20
X,Y = plt.meshgrid(np.arange(Xmin,Xmax,1/partition),np.arange(Ymin,Ymax,1/partition))
Vx,Vy = velocity(X,Y,0.1)
Vx2,Vy2 = interp(X,Y,0.1)


# vector arrows
Q = ax.quiver(X,Y,Vx,Vy,scale=10)
Q2 = ax.quiver(X,Y,Vx2,Vy2,scale=10,color='r')
Q3 = ax.quiver((Xmax-Xmin)/2, (Ymax-Ymin)/2,np.array(0), np.array(0),scale=10,color='g')

# initialize array of particles
C = np.empty([N],plt.Circle)
for i in range(0,N):
    C[i] = plt.Circle((-1,-1),radius = 0.03, fc = col[i])
    
R = np.empty([N,2],float)
for i in range(0,N):
    print("Enter x and y coordinates of the circle ",i+1)
    R[i][0] = float(input("x:"))
    R[i][1] = float(input("y:"))
    C[i].center = (R[i][0],R[i][1])
    ax.add_patch(C[i])

ani = animation.FuncAnimation(fig, animate,
         fargs=(Q,Q2,Q3,X,Y,C,R,N),
    frames = T*partition, interval=200,blit=False)

#ani.save('./Ball_demo.gif', writer='imagemagick', fps=15)   

plt.show()
