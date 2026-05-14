import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


model = Sequential()


model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(28, 28, 1)
))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu'
))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(
    filters=128,
    kernel_size=(3, 3),
    activation='relu'
))


model.add(Flatten())


model.add(Dense(128, activation='relu'))


model.add(Dropout(0.5))


model.add(Dense(10, activation='softmax'))



model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()


history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)



test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\nТочність на тестових даних: {test_acc:.4f}")



plt.figure(figsize=(12,5))


plt.subplot(1,2,1)

plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')

plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1,2,2)

plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')

plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()