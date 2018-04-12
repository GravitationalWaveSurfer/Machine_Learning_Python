############### THIS PROGRAM COMPUTES AN EXAMPLE OF LOGISTIC REGRESSSION ##################
# A dataset is read into X1,X2 which contains results for two student exams,
# and Y which contains the labels 1 and 0 for admission/rejection in a grad school.

import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model  
import scipy.optimize as opt  
from matplotlib import rc

start=time.time()

path=os.getcwd() + '/Desktop/python_alb/ML/ng3/ex2data1.txt'
data=pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Label'])
data.insert(0,'Ones',1) # add a column of ones for B 

X0=np.array(data['Ones'])
X1=np.array(data['Test1'])
X2=np.array(data['Test2'])
Y=np.array(data['Label'])
Y=np.reshape(Y,(Y.shape[0],1))
X=np.stack((X0,X1,X2)) # stacks X0,X1,X2 features in rows of new array for matrix product

####### Visualizing the data ########
plt.close('all')
Tpass=data[data['Label'].isin([1])] # creates dataframe with values labeled by y=1
Tfail=data[data['Label'].isin([0])] # same with y=0

fig1, ax = plt.subplots(figsize=(12,8))
ax.scatter(Tpass['Test1'],Tpass['Test2'], c='b', label='Pass')
ax.scatter(Tfail['Test1'],Tfail['Test2'], c='r', label='Fail')
ax.axis([0, max(X1)+10,0, max(X2)+10])
plt.show()
plt.grid('on')
plt.legend()
plt.title('Students admitted')
plt.xlabel('Grade in Test 1')
plt.ylabel('Grade in Test 2')
end=time.time()

####### Define functions for Logistic Regression #######
def sigmoid(z):
    "sigmoid activation function for logistic regression"
    sigmoid=1/(1+np.exp(-z))
    return sigmoid

def Logistic_Loss(W,X,Y):
    "Calculates logistic loss from data, weights and labels"
    loss=-np.multiply(Y,np.log(sigmoid(np.dot(W,X))))-np.multiply((1-Y),np.log(1-sigmoid(np.dot(W,X))))
    return loss    

def Logistic_Cost(W,X,Y):
    "Averages Logistic loss over samples"
    cost=np.mean(Logistic_Loss(W,X,Y))
    return cost
    
def Gradient(W,X,Y):
    "Computes gradient for a gradient descent step"
    gradient=1/Y.shape[0]*np.dot(sigmoid(np.dot(W,X))-Y.T,X.T)
    return gradient

####### Initialize learning rate and weight matrix, set number of iterations #######
alpha=0.001
W=np.zeros([1,X.shape[0]])
#W=np.array([-25,0,0])
iters=1000

def Gradient_Descent(W,X,Y,alpha,iters):
    Wnew=W.copy()
    Cost=np.zeros(iters)
    Witer=np.zeros([iters,3])
    for i in range(iters):
        Wnew=Wnew-alpha*Gradient(Wnew,X,Y)
        Cost[i]=Logistic_Cost(Wnew,X,Y)
        Witer[i]=Wnew
    return Wnew, Cost, Witer 

plt.figure(2)

for k in range(4):   
    Wnew, Cost, Witer = Gradient_Descent(W,X,Y,alpha*10**-k,iters)   
    learn=u'\u03B1 = '+ str(alpha*10**-k)
    plt.plot(Cost, '.', label=learn, markersize=3)
    

plt.show()
plt.legend()
plt.grid('on')
plt.xlabel('N of iterations')     
plt.ylabel('Cost Function') 
plt.title('Cost function evolution')    

plt.figure(3)
plt.plot(Witer)
plt.xlabel('iteration')
plt.ylabel('Weights')
plt.grid('on')
plt.title('Weight evolution')
plt.show()

Xnew=X.T
#Wnew=np.array([-25,0.222222222,0.222222222]) #why does setting this completely change accuracy?????
W_optimization = opt.fmin_tnc(func=Logistic_Cost, x0=Wnew, fprime=Gradient, args=(X, Y))  
min_cost = Logistic_Cost(W_optimization[0], X, Y)  
W_opt=np.reshape(W_optimization[0], (1,3))
def Predict_Admission(X,W_opt):
    probability=sigmoid(np.dot(W_opt,X))
    size=np.size(probability)
    Admission_result=np.zeros(size)
    print(probability)
    for l in range(size):
        if probability[0,l]>0.5:
            Admission_result[l]=1
        else: 
            Admission_result[l]=0 
    return Admission_result
    #return probability
Cand_grades=np.array([[1],[10],[10]])    
Admission_result=Predict_Admission(X,W_opt)

theta_min = np.matrix(W_optimization[0])  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(Admission_result, Y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print('accuracy = {0}%'.format(accuracy))



plt.figure(4)
x_sigm=np.linspace(-100,100,1000)
y_sigm=sigmoid(x_sigm)
plt.plot(x_sigm,y_sigm,'b')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid('on')
plt.title('Sigmoid Function')
plt.show()

print("This process ran in " + str(end-start) + " seconds")
