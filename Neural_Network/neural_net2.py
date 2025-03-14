from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Activation
import numpy as np

#_____________________________________________________________________________________
# Normalization

# MNIST-Daten laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pixel als Floats statt Integer
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Pixelwerte von [0, 255] zu [0, 1]
x_train /= 255
x_test /= 255

# Überprüfung, ob Normalisierung korrekt
if x_train.dtype != np.float32 or x_test.dtype != np.float32:
    raise Exception("Einer der x-Arrays hat den falschen Datentyp!")

if max(x_train.max(), x_test.max()) > 1 or min(x_train.min(), x_test.min()) < 0:
    raise Exception("Einer der x-Arrays wurde nicht richtig normalisiert!")

# Letzte Dimension als Channeldimension (hier nur ein Channel)
x_train = x_train.reshape((x_train.shape[0], 28, 28, -1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, -1))

#_____________________________________________________________________________________
# Add channel dimension

# Überprüfung, ob Dimensionen korrekt
if x_train.shape != (x_train.shape[0], 28, 28, 1):
    raise Exception("x_train wurde nicht richtig dimensioniert!")

if x_test.shape != (x_test.shape[0], 28, 28, 1):
    raise Exception("x_test wurde nicht richtig dimensioniert!")

#_____________________________________________________________________________________
# Transformation of y data
from tensorflow.keras.utils import to_categorical

# Label zu kategorischem Format (One-Hot) umformen
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#_____________________________________________________________________________________
# NN model definition
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout

# 1. sequentielles Modell erstellen und Input-Layer hinzufügen
model = Sequential()
model.add(Input(shape=(28, 28, 1)))

# 2. Convolutional Layer hinzufügen
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))

# 3. MaxPool-Layer hinzufügen
model.add(MaxPooling2D(pool_size=(2, 2)))

# 6. Dropout-Layer mit Rate 0.25 hinzufügen
model.add(Dropout(0.25))

# 4. Flatten- und Dense-Layer hinzufügen
model.add(Flatten())
model.add(Dense(128, activation="relu"))

# 6. Dropout-Layer mit Rate 0.5 hinzufügen
model.add(Dropout(0.5))

# 5. Output-Layer hinzufügen
model.add(Dense(10, activation="softmax"))

# 7. summary analysieren
model.summary()

#_____________________________________________________________________________________
# Training

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

nr_samples = 3000
model.fit(x_train[:nr_samples], y_train[:nr_samples], batch_size=64, epochs=5, validation_data=(x_test, y_test))

#_____________________________________________________________________________________
# Testing
from matplotlib import pyplot as plt

nr_examples = 3

# wähle zufällige Indizes der Beispiele
indexes = np.random.randint(x_test.shape[0], size=nr_examples)

# wähle Beispiele und inferiere labels
examples = x_test[indexes,:]
labels = np.argmax(model.predict(examples), axis=-1)

# gib jeweils Inferenz und Bild aus
for image, prediction in zip(examples, labels):
    print(f"\nFolgende Zahl ist eine {prediction}:")
    plt.imshow(image, cmap="Greys")
    plt.show()
