from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy

(X_train, y_train), (X_test, y_test) = mnist.load_data()

for i in range(5):
    plt.imshow(X_train[i*40], cmap='gray')
    plt.show()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

classes = 10
y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)

input_size = 784
batch_size = 100
hidden_neurons = 100
epochs = 100

model = Sequential([
    Dense(hidden_neurons, input_dim=input_size),
    Activation('sigmoid'),
    Dense(classes),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
score = model.evaluate(X_test, y_test, verbose=1)
print('Test accuracy:', score[1])
weights = model.layers[0].get_weights()

fig = plt.figure()
w = weights[0].T
for neuron in range(hidden_neurons):
    ax = fig.add_subplot(10, 10, neuron + 1)
    ax.axis("off")
    ax.imshow(numpy.reshape(w[neuron], (28, 28)), cmap=cm.Greys_r)

plt.savefig("neuron_images.png", dpi=300)
plt.show()

y_pred = model.predict(X_test[:16])
print("Predito: ", y_pred)
print("Real: ", y_test[:16])