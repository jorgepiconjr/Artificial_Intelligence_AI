import matplotlib.pyplot as plt
import numpy as np
import math

def f(x):
    return w1*x+w0
w1 = 2
w0 = 2

def logistic(x):
    return 1/(1+math.exp(-x))
def logistic_dx(x):
    return logistic(x) * (1-logistic(x))

p1 = (-1,7)
p2 = (1,3.5)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)

plt.ylim(-6,10)
plt.xlim(-5,5)
plt.scatter([-2, -4, 0.5],[2, 4, 8])
plt.scatter([-2, 2, 3],[-5, 4, -2], marker="x")
plt.scatter([p1[0],p2[0]],[p1[1],p2[1]],marker="*", color="red")
plt.annotate("P1",(p1[0]+0.1,p1[1]+0.1))
plt.annotate("P2",(p2[0]+0.1,p2[1]-0.6))
plt.plot([p1[0],p1[0]],[f(-1),p1[1]], color="red")
plt.plot([p2[0],p2[0]],[f(1),p2[1]], color="red")
plt.plot([-6,6],[f(-6),f(6)], label="f(x)", color = "green")
plt.ylabel("y")
plt.xlabel("x")
plt.legend()
plt.grid(color="lightgrey", linewidth=0.5)

dist_p1 = f(p1[0]) - p1[1]
dist_p2 = f(p2[0]) - p2[1]
print("\nDistanz P1 zu f(x):", dist_p1)
print("Distanz P2 zu f(x): ", dist_p2, end = "\n\n\n")

plt.subplot(1,2,2)

ax = plt.subplot(1, 2, 2)

ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

xs = np.arange(-8,8,0.1)
plt.plot(xs,[logistic(x) for x in xs], label="$logistic(x)=1/(1+e^{-x})$")
plt.scatter([dist_p1],[logistic(dist_p1)],marker="*", color="red")
plt.annotate("P1",(dist_p1,logistic(dist_p1)+0.03))
plt.scatter([dist_p2] ,[logistic(dist_p2)],marker="*", color="red")
plt.annotate("P2",(dist_p2+0.3,logistic(dist_p2)))
plt.ylabel("y")
plt.xlabel("x")
plt.legend(loc=2)
plt.show()

print("\nAuswertungen durch die logistische Funktion:")
print("P1: Entfernung =", dist_p1, ", logistic(Entfernung) =", logistic(dist_p1), "-> Klasse 0... blau")
print("P2: Entfernung = ", dist_p2, ", logistic(Entfernung) =", logistic(dist_p2), "-> Klasse 1... orange")


