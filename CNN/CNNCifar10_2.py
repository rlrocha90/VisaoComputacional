import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batch_size = 50

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
for i in range(5):
    plt.imshow(X_train[i])
    plt.show()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)

# Aumento de Dados
data_generator = ImageDataGenerator(rotation_range=90,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    horizontal_flip=True)
data_generator.fit(X_train)

# Padronização dos dados de teste
for i in range(len(X_test)):
    X_test[i] = data_generator.standardize(X_test[i])

# Definição do modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:])) # 32 filtros, 3x3, stride 1 padding...
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(32, (3, 3), padding='same')) # 32 filtros, 3x3, stride 1 padding...
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same')) # 64 filtros, 3x3, stride 1 padding...
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(64, (3, 3), padding='same')) # 64 filtros, 3x3, stride 1 padding...
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3))) # 128 filtros, 3x3
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(128, (3, 3))) # 128 filtros, 3x3
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=data_generator.flow(
    x=X_train,
    y=Y_train,
    batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=5,
    validation_data=(X_test, Y_test),
    workers=4)

y_pred = model.predict(X_test)
print(y_pred)
print(Y_test)