import numpy as np

#_____________________________________________________________________________________
### Layer Class
class Layer:
    """
    Eine abstrakte Klasse, die die Funktionen eines Layers im neuronalen Netz definiert.
    """
    def __init__(self):
        """
        Initialisiert self.input und self.output leer.
        """
        self.input = None
        self.output = None

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Berechnet Output Y des Layers für gegebenen Input X.
        Sollte Input in self.input und Output in self.output speichern (wird meist für Backpropagation gebraucht)!
        :param input_data: Input X
        :return: Output Y
        """
        raise NotImplementedError("Subklassen müssen diese Funktion implementieren! Die Layer-Klasse ist abstrakt.")

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Berechnet Input-Fehler dE/dX für einen gegebenen Output-Fehler dE/dY und aktualisiert Parameter, falls vorhanden.
        :param output_error: Output-Fehler dE/dY
        :param learning_rate: Lernrate a
        :return: Input-Fehler dE/dX
        """
        raise NotImplementedError("Subklassen müssen diese Funktion implementieren! Die Layer-Klasse ist abstrakt.")
#_____________________________________________________________________________________
### Fully Connected Layer
class FCLayer(Layer):
    """
    Ein Layer, bei dem jedes Input-Neuron mit jedem Output-Neuron verbunden ist.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initialisiert zufällige weights W und biases B abhängig von Input- und Output-Größe des Layers.
        :param input_size: Anzahl an Input-Neuronen m
        :param output_size: Anzahl an Output-Neuronen n
        """
        super().__init__()

        # initialisiere zufällige weights W
        # W hat Dimension m x n
        # Zufallswerte in [-0.5, 0.5)
        self.weights = np.random.rand(input_size, output_size) - 0.5

        # initialisiere zufällige biases B
        # B hat Dimension 1 x n
        # Zufallswerte in [-0.5, 0.5)
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Berechnet Output Y des Layers für gegebenen Input X.
        :param input_data: Input X
        :return: Output Y
        """
        # speichere Input X (wird für Backpropagation gebraucht)
        self.input = input_data
        # berechne Output mit Y = X * W + B
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Berechnet Input-Fehler dE/dX für einen gegebenen Output-Error dE/dY und aktualisiert Parameter.
        :param output_error: Output-Fehler dE/dY
        :param learning_rate: Lernrate
        :return: Input-Fehler dE/dX
        """
        # berechne Input-Fehler als dE/dX = dE/dY * W^T
        input_error = np.dot(output_error, self.weights.T)

        # berechne Weight-Fehler als dE/dW = X^T * dE/dY
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        # W = W - alpha * dE/dW
        self.weights -= learning_rate * weights_error
        # B = B - alpha * dE/dY
        self.bias -= learning_rate * output_error
        return input_error


#_____________________________________________________________________________________
### Activation Layer
class ActivationLayer(Layer):
    """
    Ein Layer, der eine gegebene Aktivierungsfunktion auf den Input anwendet.
    """
    def __init__(self, activation, activation_prime):
        """
        Initialisiert die verwendete Aktivierungsfunktion und ihre Ableitung.
        :param activation: Aktivierungsfunktion f
        :param activation_prime: Ableitung f' der Aktivierungsfunktion f
        """
        super().__init__()
        # initialisiere self.activation
        self.activation = activation
        # initialisiere self.activation_prime
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Berechnet Output Y des Layers für gegebenen Input X.
        :param input_data: Input X
        :return: Output Y
        """
        # speichere Input X (wird für backpropagation gebraucht)
        self.input = input_data
        # berechne output mit Y = f(X)
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Berechnet Input-Fehler dE/dX für einen gegebenen Output-Error dE/dY.
        :param output_error: Output-Fehler dE/dY
        :param learning_rate: Lernrate
        :return: Input-Fehler dE/dX
        """
        # Layer hat keine Parameter, die aktualisiert werden können
        # dE/dX = dE/dY o f'(X)
        return output_error * self.activation_prime(self.input)

### Activation function: tanh

def tanh(x: np.ndarray) -> np.ndarray:
    """
    Berechnet tanh-Funktion
    :param x: Argument x
    :return: Bild y = tanh(x)
    """
    return np.tanh(x)

def tanh_prime(x: np.ndarray) -> np.ndarray:
    """
    Berechnet Ableitung der tanh-Funktion
    :param x: Argument x
    :return: Bild y = tanh'(x)
    """
    return 1 - np.tanh(x) ** 2


#_____________________________________________________________________________________
### Mean Squared Error (MSE)
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Berechnet Mean Squared Error.
    :param y_true: Matrix Y* mit korrekten Werten (Gold standard)
    :param y_pred: Matrix Y mit berechnetem Ergebnis
    :return: MSE(Y*, Y)
    """
    return np.mean(np.power(y_pred - y_true, 2))

def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Berechnet Ableitung dE/dY des Mean Squared Error.
    :param y_true: Matrix Y* mit korrekten Werten (Gold standard)
    :param y_pred: Matrix Y mit berechnetem Ergebnis
    :return: MSE'(Y*, Y)
    """
    return 2 / y_true.size * (y_pred - y_true)


#_____________________________________________________________________________________
### Network Class
from tqdm import tqdm

class Network:
    """
    Eine Klasse, die eine Liste von Layern und eine Verlustfunktion beinhaltet und mit deren Hilfe Forward- und Backpropagation ermöglicht.
    """
    def __init__(self):
        """
        Initialisiert eine leere Liste an Layern in self.layers, sowie self.loss und self.loss_prime.
        """
        # initialisiere self.layers
        self.layers = []
        # initialisiere self.loss
        self.loss = None
        # initialisiere self.loss_prime
        self.loss_prime = None

    def add_layer(self, layer: Layer) -> None:
        """
        Fügt den gegebenen Layer zum Netzwerk hinzu.
        :param layer: Layer, der hinzugefügt werden soll.
        :return:
        """
        self.layers.append(layer)

    def use_loss_function(self, loss, loss_prime) -> None:
        """
        Verändert die genutzte Verlustfunktion E.
        :param loss: Verlustfunktion E.
        :param loss_prime: Ableitung dE/dY der Verlustfunktion.
        :return:
        """
        # setze self.loss
        self.loss = loss
        # setze self.loss_prime
        self.loss_prime = loss_prime

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Berechnet den Output Y des Netzes für jeden gegebenen Input X.
        :param input_data: Liste von Inputs X
        :return: Liste von Outputs Y
        """
        result = []

        # berechne Output für jeden Input X in input_data
        # tqdm erstellt automatische Progressbar
        for input_element in tqdm(input_data, desc='Prediction'):
            # Output wird als input X initialisiert
            output = input_element

            # Output wird von Layer zu Layer propagiert
            for layer in self.layers:
                output = layer.forward_propagation(output)

            # Output wird an Liste von Outputs angehangen
            result.append(output)

        # Funktion gibt numpy-Array mit Outputs zu den jeweiligen Inputs zurück
        return np.array(result)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float) -> None:
        """
        Trainiert das Netz auf den gegebenen Daten.
        :param x_train: Liste von Inputs X
        :param y_train: Liste von richtigen Outputs Y*
        :param epochs: Anzahl Wiederholungen des Trainings
        :param learning_rate: Lernrate für das Training
        :return:
        """

        # Training wird so oft wiederholt, wie in epochs angegeben
        # tqdm erstellt automatische Progressbar
        for _ in (process_bar := tqdm(range(epochs), desc='Training')):
            # Fehler dieser Epoche wird mit 0 initialisiert
            epoch_error = 0

            # Netz wird auf allen Input-Output-Paaren trainiert
            for x_value, y_value in zip(x_train, y_train):
                # Output wird als Input X initialisiert
                output = x_value
                # Output wird von Layer zu Layer propagiert
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Fehler beim aktuellen Input-Output-Paar wird epoch_error hinzugefügt
                epoch_error += self.loss(y_value, output)

                # nach hinten propagierter Fehler wird als dE/dY initialisiert
                propagated_error = self.loss_prime(y_value, output)

                # Fehler wird (rückwärts) von Layer zu Layer propagiert
                for layer in reversed(self.layers):
                    propagated_error = layer.backward_propagation(propagated_error, learning_rate)

            # durchschnittlicher Fehler der Epoche wird gebildet
            epoch_error /= len(x_train)
            # Progressbar wird mit durchschnittlichem Fehler der Epoche aktualisiert
            process_bar.set_postfix_str(f'Aktueller Fehler: {epoch_error}', refresh=False)


#_____________________________________________________________________________________
#_____________________________________________________________________________________
#_____________________________________________________________________________________
### MNIST
from tensorflow.keras.datasets import mnist
from keras import utils as keras_utils

def reshape(x_data, y_data):
    x_data = x_data.reshape(x_data.shape[0], 1, 28 * 28)
    x_data = x_data.astype(np.float32)
    x_data /= 255
    y_data = keras_utils.to_categorical(y_data)
    return x_data, y_data


# lade MNIST-Daten
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalisiere Daten
x_train, y_train = reshape(x_train, y_train)
x_test, y_test = reshape(x_test, y_test)

# erstelle Netz
net = Network()
net.add_layer(FCLayer(28 * 28, 100))
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FCLayer(100, 50))
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FCLayer(50, 10))
net.add_layer(ActivationLayer(tanh, tanh_prime))

#________________________________________________
### Training
# Verlustfunktion festlegen
net.use_loss_function(mse, mse_prime)

nr_samples = 3000   # @param {type:"integer"}
epochs = 30         # @param {type:"integer"}
learning_rate = 0.1 # @param {type:"slider", min:0.1, max:1, step:0.1}

# trainiere auf ersten `nr_samples` Beispielen
net.fit(x_train[:nr_samples], y_train[:nr_samples], epochs=epochs, learning_rate=learning_rate)

#________________________________________________
###  Accurary check
def max_val_index(array: np.ndarray) -> int:
    return max(enumerate(array), key=(lambda x: x[1]))[0]

# teste prediction
out = net.predict(x_test)

nr_matches = sum([max_val_index(expected) == max_val_index(actual) for expected, [actual] in zip(y_test, out)])

accuracy = nr_matches / len(y_test)
print(f'Accuracy: {round(accuracy * 100, ndigits=2)}%')