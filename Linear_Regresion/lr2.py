import numpy as np
import matplotlib.pyplot as plt
def update2(x, a, b, c, d, alpha):
    x = x - alpha*(4*a*x**3 + 2*b*x + c)

    return x

def V(x, a, b, c, d):
    return a*x**4 + b*x**2 + c*x + d

a = 1
b = -3
c = 1
d = 3.514

x0 = np.array([-1.75, -1.75, -1.75, -1.75])
iterations = 101
alphas = np.array([0.001, 0.19, 0.205, 0.02])

losses = np.empty(shape=(iterations, len(alphas)))
results = np.empty(len(alphas))

for j in range(len(alphas)):
    x = x0[j]
    alpha = alphas[j]
    for i in range(iterations):
        losses[i, j] = V(x, a, b, c, d)
        if i != iterations - 1:
            x = update2(x, a, b, c, d, alpha)
    results[j] = x

for j in range(len(alphas)):
    print(100*"-")
    print("Alpha: ", alphas[j])
    print("xmin: ", results[j])
    print("Loss: ", V(results[j], a, b, c, d))

colors = ["blue", "red", "black", "orange", "green", "grey"]

plt.figure(figsize=(8, 8))
plt.title("Lernkurven")
plt.xlabel("Epoche")
plt.ylabel("Loss V")
plt.xlim(0, iterations)

for i in range(len(alphas)):
    alpha = alphas[i]
    plt.plot(range(iterations), losses[:, i], label="(" + str(x0[i]) + " " + str(alpha) + ")", color=colors[i])

plt.legend()
plt.ylim(bottom=0)
plt.show()

plt.figure(figsize=(8, 8))
plt.title("Funktion V und Minima")
plt.xlabel("x")
plt.ylabel("V(x)")

xs = np.linspace(-2, 2, 100)
ys = V(xs, a, b, c, d)

plt.plot(xs, ys)

# Visualization gradient descent method
for j in range(len(alphas)):
    alpha = alphas[j]
    xmin = results[j]
    vxmin = V(xmin, a, b, c, d)
    plt.plot(xmin, vxmin, marker='.', linestyle="None", label="(" + str(x0[j]) + " " + str(alpha) + ")", color=colors[j], ms=10)
plt.legend()
plt.show()