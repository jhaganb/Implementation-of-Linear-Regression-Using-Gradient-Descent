# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Jhagan B
RegisterNumber:  212220040066
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))

```
## Output:
Profit prediction:

![image](https://github.com/jhaganb/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/63654882/dadbe887-e775-4cc0-ac06-038bd2a16ccc)

Function:

![image](https://github.com/jhaganb/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/63654882/104d20d2-bb34-458b-abe4-dae78d341410)

GRADIENT DESCENT:

![image](https://github.com/jhaganb/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/63654882/1cf7956a-264f-4843-9f5b-1701e57987c8)

COST FUNCTION USING GRADIENT DESCENT:

![image](https://github.com/jhaganb/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/63654882/beeab19d-57e8-43e4-9601-30b1b3530cb9)

LINEAR REGRESSION USING PROFIT PREDICTION:

![image](https://github.com/jhaganb/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/63654882/44cc7c8c-05c2-42e9-aade-6cbc5b332f07)

PROFIT PREDICTION FOR A POPULATION OF 35000:

![image](https://github.com/jhaganb/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/63654882/5d53c5b1-0886-4115-80d1-cb595ca1b5e3)

PROFIT PREDICTION FOR A POPULATION OF 70000:

![image](https://github.com/jhaganb/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/63654882/bb377991-472d-4a7b-a72b-cc3b50e5ea4a)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
