#!/usr/bin/env python
# coding: utf-8

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train,num_classes=10)
    y_test = to_categorical(y_test,num_classes=10)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return x_train, y_train, x_test, y_test


def build_lenet5_model():
    # Building model
    model = Sequential()
    # YOU CODE HERE
	
    # First Convolutional Layer with Tanh Activation
    model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='tanh'))
    # First Pooling Layer
    model.add(MaxPool2D(pool_size=(2,2)))	

    # Second Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
    # Second Pooling Layer
    model.add(MaxPool2D(pool_size=(2,2)))

    # Fully-connected Layer
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))

    # Classifier Layer
    model.add(Dense(10, activation='softmax'))    

    return model


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist_data()
    model = build_lenet5_model()
    print(model.summary())
    # Compiling model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.1),
        metrics=['accuracy']
    )

    # Training model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128)

    # Evaluate model
    score = model.evaluate(x_train, y_train)
    print("Total loss on Training Set:", score[0])
    print("Accuracy of Training Set:", score[1])
     
    score = model.evaluate(x_test, y_test)
    print("Total loss on Testing Set:", score[0])
    print("Accuracy of Testing Set:", score[1])
