import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Softmax, Dropout
from tensorflow.python.keras.layers import UpSampling2D
from PIL import Image


class AutoencoderManu:

    def __init__(self, plot=True, loss='sparse_categorical_crossentropy', metrics=['accuracy']):

        # Setting dataset, checkpoint and other parameters
        self.dataset = keras.datasets.cifar10
        self.checkpoint_path = "cp_manu/cp.ckpt"
        self.opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
        self.plot = plot

        self.model = Sequential()

        # Setting Encoder architecture
        self.encoder = Sequential()
        self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # 3x32x32 -> 32x16x16
        self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.encoder.add(MaxPool2D((2, 2), padding='same'))

        self.encoder.add(Dropout(0.2))
        self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 32x16x16 -> 64x8x8
        self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.encoder.add(MaxPool2D((2, 2), padding='same'))

        self.encoder.add(Dropout(0.2))
        self.encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 128x4x4
        self.encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))


        # Setting Decoder architecture
        self.decoder = Sequential()

        self.decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 128x4x4 -> 64x8x8
        self.decoder.add(UpSampling2D((2, 2)))

        self.decoder.add(Dropout(0.2))
        self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 32x16x16
        self.decoder.add(UpSampling2D((2, 2)))

        self.decoder.add(Dropout(0.2))
        self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # 32x16x16 -> 3x32x32

        # Setting Autoencoder architecture
        self.autoencoder = Sequential()
        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)

        # Setting Classifier architecture
        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(64, activation='relu'))
        self.classifier.add(Dense(10, activation='softmax'))

        # Setting Model architecture
        self.model.add(self.encoder)
        self.model.add(self.classifier)

        # Compiling the model and the autoencoder
        self.model.compile(optimizer=self.opt, loss=loss, metrics=metrics)
        self.autoencoder.compile(optimizer=self.opt, loss=tf.keras.losses.MeanSquaredError(), metrics=metrics)

    def train(self, epochs=3):

        # Setting up the checkpoint
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        # Training the model
        (train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()
        print(train_images.shape)
        history = self.model.fit(train_images, train_labels,
                                 epochs=epochs,
                                 validation_data=(test_images, test_labels),
                                 callbacks=[cp_callback])

        # Plotting the results
        if self.plot:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.show()

    def train_autoencoder(self, epochs=3):

        # Setting up the checkpoint
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        # Training the autoencoder
        (train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()
        train_images = train_images.astype('float32') / 255.
        test_images = test_images.astype('float32') / 255.
        history = self.autoencoder.fit(train_images, train_images,
                                       epochs=epochs,
                                       validation_data=(test_images, test_images),
                                       callbacks=[cp_callback])

        # Plotting the results
        if self.plot:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.show()

    def encode(self, image):
        (train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()
        print(train_images[0])

        train_images = train_images.astype('float64') / 255.
        test_images = test_images.astype('float64') / 255.

        decoded_imgs = self.autoencoder.predict(train_images[:5])

        # display reconstruction
        plt.subplot(211)
        plt.imshow(train_images[4].reshape(32, 32, 3))

        plt.subplot(212)
        plt.imshow(decoded_imgs[4].reshape(32, 32, 3))
        plt.show()









