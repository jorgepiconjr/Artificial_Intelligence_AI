import numpy as np
import matplotlib.pyplot as plt

plt.ylim(0,350)
plt.xlim(0,160)
plt.scatter([30, 50, 70, 90, 120, 150], [70, 80, 105, 140, 210, 300])
plt.plot([0, 160],[12.5, 319.1666], color='grey', linestyle='dashed', linewidth=2)
plt.xlabel("v")
plt.ylabel("c")
plt.show()

def update(w1, w0, alpha, v, c):
  n = len(v)
  dw1 = 1/n*np.sum((w0 + w1*v - c)*v)
  dw0 = 1/n*np.sum(w0 + w1*v - c)

  w1 = w1 - alpha*dw1
  w0 = w0 - alpha*dw0

  return w1, w0


v = np.array([30, 50, 70, 90, 120, 150])
c = np.array([70, 80, 110, 140, 200, 300])

iterations = 100

# Startwerte
w1 = 2
w0 = 2

# Lernrate
alpha = 0.0001

for i in range(iterations):
  w1, w0 = update(w1, w0, alpha, v, c)

print(f"Numerische Lösung: c = {w1:.2f}v + {w0:.2f}")

# Optimale Lösung (Aufgabe 2.4)
bestw1 = (np.mean(c*v) - np.mean(c)*np.mean(v))/(np.mean(v**2) - np.mean(v)**2)
bestw0 = np.mean(c) - bestw1*np.mean(v)

print(f"Optimale Lösung: c = {bestw1:.2f}v + {bestw0:.2f} (grün)")

plt.figure()
plt.xlabel(r"$v/\frac{km}{h}$")
plt.ylabel(r"$c/\frac{Wh}{km}$")
plt.plot(v, c, '.')
plt.plot(v, v*w1 + w0)
plt.plot(v, v*bestw1 + bestw0, color="green") # Optimale Lösung (Aufgabe 2.4)
plt.show()