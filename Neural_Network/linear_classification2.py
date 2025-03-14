'''  Regression vs. Classification
The code below implements gradient descent for linear regression on some
sample data. However, in addition to the x and y coordinates, the data
also has a class label.
'''
import math
import random
import matplotlib.pyplot as plt

data = [(0.44, 0.12, 0), (1.15, 0.15, 0), (1.46, 0.34, 0), (1.5, 0.35, 0), (2.35, 0.57, 0),
        (2, 0.72, 0), (6.8, 0.91, 1), (3.98, 0.86, 1), (4.1, 0.98, 1), (4.92, 0.79, 1),
        (5.29, 1.22, 1), (6.31, 1.29, 1), (6.09, 1.29, 1), (6.5, 1.52, 1), (7.54, 1.19, 1),
        (8.47, 1.43, 1), (8.33, 1.36, 1), (8.75, 1.49, 1), (9.69, 2.0, 1), (9.8, 1.65, 1)]

x1,x2 = 0,10
n = len(data)

def f(x):
    return w1*x + w0

# The mean squared error
# Takes a list of the true/expected values and a list of the results obtained by our model
def loss(ys, fs):
    return (1.0/n) * sum([math.pow(y - f, 2) for (y,f) in zip(ys, fs)])

### Data Point visualitation

def showplot():
    plt.ylim(0,2.5)
    plt.xlim(0,10)
    plt.scatter([x for (x,y,z) in data if z],[y for (x,y,z) in data if z])
    plt.scatter([x for (x,y,z) in data if not z],[y for (x,y,z) in data if not z])
    plt.ylabel("y")
    plt.xlabel("x")
    plt.show()

showplot()

### Gradient Descent for Linear Regression

w0,w1=0.0,0.0
epochs = 10
alpha = 0.004
for i in range(epochs):
    plt.plot([x1,x2], [f(x1),f(x2)], color=(1,0,0,(0.05+0.95*(i+1)/epochs)))  # regression line
    w0 += alpha*(2.0/n)*sum([  (y-f(x)) for (x,y,z) in data])
    w1 += alpha*(2.0/n)*sum([x*(y-f(x)) for (x,y,z) in data])

print(f"Durch Gradientenabstieg in {epochs} Schritten gefundene Lösung")
print(f"f(x)= {w1:.2f} x + {w0:.2f}")
print(f"Loss: {loss([y for (x,y,z) in data],[f(x) for (x,y,z) in data]):.2f}")

showplot()

### Gradient Descent to Train Linear Classifier
def logistic(x,y):
    return 1/(1+math.exp(-(w0+w1*x-y)))

w0,w1=0.0,0.0
epochs = 24
alpha = 2.5
for i in range(epochs):
    plt.plot([x1,x2], [f(x1),f(x2)], color=(1,0,0,(0.05+0.95*(i+1)/epochs)))  # regression line
    w0 += alpha*(2.0/n)*sum([(z-logistic(x,y))*logistic(x,y)*(1-logistic(x,y)) for (x,y,z) in data])
    w1 += alpha*(2.0/n)*sum([(z-logistic(x,y))*logistic(x,y)*(1-logistic(x,y))*x for (x,y,z) in data])

print(f"Durch Gradientenabstieg in {epochs} Schritten gefundene Lösung:\n f(x)= {w1:.2f} x + {w0:.2f}")
print(f"Die logistische Funktion, mit deren Hilfe klassifiziert wird, ist entsprechend:\n label(x,y)=1/(1+e^(-({w0:.2f}+{w1:.2f}*x-y)))")
print(f"Loss: {loss([z for (x,y,z) in data],[logistic(x,y) for (x,y,z) in data]):.2f}")

showplot()

print(f"Label von (0,0) ist: {int(round(logistic(0,0),0))}")
print(f"Label von (6,0.5) ist: {int(round(logistic(6,0.5),0))}")