#Import all the required libraries here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#define the join function
def join_func(X, g, y):
    return -1*( X.T @ (g - y)) 

#I define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#to normalise X 
def normalize(arr, temp):
    mean = np.mean(arr)
    std = np.std(arr)
    median = np.median(arr)
    arr = (arr-mean)/std  #z-score method to normalise the data 
    return arr

#hession matrix H as for newtons method we compute the hessian for the double derivatives of gradients
## follow : https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function for derivation  
def H(X, g):
    diagnol_val = np.array([i*(1-i) for i in g])
    diagnol_matrix = np.diag(diagnol_val)
    Temp = (X.T @ diagnol_matrix @ X)    # H⃗ (ω)=∇⃗ 2l(ω)=XDXT
    H = (-1)*Temp
    return H

#Logistic function 
def logistic(X,y,theta, episilon):
    flag = 1
    for i in range(0,1000):
        g = sigmoid(X@theta) #the link function  1/1+e^theta.TX
        join = join_func(X,g, y) #the hypothesis funtion we choose 
        Hessian = H(X,g)
        H_ = np.linalg.inv(Hessian)
        theta = theta - H_ @ join # Qt+1= Qt- [H]^-1.h0
        for j in join:
            if np.abs(j) > episilon: #if the value of join function at jth index is more than episilon the algo continues to find the maxima
                flag = 0
        if(flag !=0):
            break
        flag = 1    # the algo finds the maxima and breaks out of the loop 
    print('Theta :', theta)
    print('Iterations# ',i)
    slope = (theta[1]/theta[2])
    intercept = (theta[0]/theta[2])
    separator(slope,intercept)
    plt.xlabel('X1 ')
    plt.ylabel('X2 ')
    print(X.shape)
    #initialise then as empty array 
    xdata1=[]
    ydata1=[]
    xdata2=[]
    ydata2=[]
    #iterate and fill the values of data points, these arrays will be used to plot the linear separator and generate the data 
    for i in range(0,100):
        if y[i] == 0:
            xdata1.append(X[i][1]) 
            ydata1.append(X[i][2]) 
        else :
            xdata2.append(X[i][1]) 
            ydata2.append(X[i][2])
   
    plt.title('Descison boundary using newtowns method')
    plt.scatter(xdata1,ydata1,s=20,c='b',label='Label0')
    plt.scatter(xdata2,ydata2,s=20,c='g',label='Label1')
    plt.legend()
    plt.show()
    
#It draws the linear boundary in the data
def separator(slope, intercept):
    x = np.linspace(-3,5,10)
    y_ = -1*intercept + -1*slope * x #lineae formula y= mx+c
    plt.plot(x, y_, 'r-', label = "Linear Separator")
    legend = plt.legend(loc="upper left")

def main_func():
    #Read input files
    input_X = "logisticX.csv"
    input_Y = "logisticY.csv" 
    X_data = np.loadtxt(input_X,delimiter = ',') 
    Y_data = np.loadtxt(input_Y)
    y = Y_data
    #initializing the parameters 
    m,n =X_data.shape
    theta = [0,0,0]
    ones_list = np.ones(m)
    #normalise the data by calling the function 
    X0 = normalize(X_data.T[0],0)
    X1 = normalize(X_data.T[1],0)
    
    #ones are added to the x matrix
    X = np.stack((ones_list,X0, X1), axis=-1)
    
    #stoping criteria
    episilon = 0.0000000000001 #10^-12

    logistic(X,y, theta, episilon ) 
main_func()
