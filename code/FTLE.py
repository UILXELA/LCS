# Jialun Bao
# 11/13/15

import numpy as np
import numpy.linalg as LA
import time as time

# constants
L = 400
H = 200
delta = 0.01

start_time = time.time()

# spatial Jacobian that is used to compute FTLE
def Jacobian(X,Y):
    J = np.empty([2,2],float)
    FTLE = np.empty([H-2,L-2],float)
    
    for i in range(0,H-2):
        for j in range(0,L-2):
            J[0][0] = (X[(1+i)*L+2+j]-X[(1+i)*L+j])/(2*delta)
            J[0][1] = (X[(2+i)*L+1+j]-X[i*L+1+j])/(2*delta)
            J[1][0] = (Y[(1+i)*L+2+j]-Y[(1+i)*L+j])/(2*delta)
            J[1][1] = (Y[(2+i)*L+1+j]-Y[i*L+1+j])/(2*delta)
			
			# Green-Cauchy tensor
            D = np.dot(np.transpose(J),J)
			# its largest eigenvalue
            lamda = LA.eigvals(D)
            FTLE[i][j] = max(lamda)
    return FTLE

# save 100 FTLE fields from 0-10s
op=input("Please type in 0 to generate mapping files for model, 1 for data\n")
for i in range(0,100):
    if(op=="0"):
        Input = open('./mapping_files_model/mapping%d.txt'%i,'r')
    elif(op=="1"):
        Input = open('./mapping_files/mapping%d.txt'%i,'r')
    X,Y = np.loadtxt(Input,unpack=True)
    Input.close()
    FTLE = Jacobian(X,Y)
    FTLE = np.log(FTLE)
    if(op=="0"):
        np.savetxt('./FTLE_files_model/FTLE%d.txt'%i,FTLE)
    elif(op=="1"):
        np.savetxt('./FTLE_files/FTLE%d.txt'%i,FTLE)
    

print(time.time()-start_time)

