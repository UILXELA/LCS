import pylab as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd

# constants
p = np.pi
A = 0.1
epsilon = 0.25
w = p/5
delta = 0.0001
dt = 0.1*0.5
partition = 20

#!!!Must be the smae through all files!!!
Xmin=0
Xmax=2
Ymin=0
Ymax=1
T=200

# wave function that defines the characteristics of
# double gyre
def phi(x,y,t):
    temp = A*np.sin(p*f(x,t))*np.sin(p*y)
    return temp

def f(x,t):
    temp = epsilon*np.sin(w*t)*x**2+(1-2*epsilon*np.sin(w*t))*x
    return temp

fig, ax = plt.subplots(1,1,figsize=(10,5))

def velocity(x,y,t):
    vx = (phi(x,y+delta,t)-phi(x,y-delta,t))/(2*delta)
    vy = (phi(x-delta,y,t)-phi(x+delta,y,t))/(2*delta)
    return vx,vy

def txtout(vx,vy,t):
    np.savetxt("./txt_data/X_Velocity %.2f.txt"%t,vx,delimiter=', ', newline='\n',)
    np.savetxt("./txt_data/Y_Velocity %.2f.txt"%t,vy,delimiter=', ', newline='\n',)

def csvout(vx,vy,t):
    x = pd.DataFrame(vx)
    y = pd.DataFrame(vy)
    x.to_csv("./csv_data/X_Velocity %.2f.csv"%t,index=False,header=False)
    y.to_csv("./csv_data/Y_Velocity %.2f.csv"%t,index=False,header=False)
def data_gen(x_min,x_max,y_min,y_max,T,op): #x_min, x_max, y_min, y_max are the range of the grid. T is time. 
    X,Y = plt.meshgrid(np.arange(x_min,x_max,1/partition),np.arange(y_min,y_max,1/partition))
    size = [2,X.shape[0],X.shape[1],(int)(T/dt)+1]
    data = np.zeros(size)
    op=int(op)
    if (op!=1 and op!=2):
        print(op)
        print("Unexpected Input \n Please type in 1 for csv generation or 2 for txt generation")
        return

    print(X.shape)
    for t in np.arange(0,T+1,dt):
        print(t)
        vx,vy = velocity(X,Y,t)
        if(op==1):
            csvout(vx,vy,t)
        elif(op==2):
            txtout(vx,vy,t)





option = input("Please type in 1 for csv generation or 2 for txt generation\n")
data_gen(Xmin,Xmax,Ymin,Ymax,T,option)
