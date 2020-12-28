import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Softmax, Dropout, Reshape
from tensorflow.python.keras.layers import UpSampling2D
from PIL import Image


class AutoencoderManu:

    def __init__(self, folder='', topology=1, plot=True, loss='sparse_categorical_crossentropy'):

        # This parameter is used to keep track of which autoencoder we are training
        self.id_topology = topology

        # Setting dataset, checkpoint and other parameters
        self.latent_dimension = 2048
        self.dataset = keras.datasets.cifar10
        self.checkpoint_path = "cp_manu/" + folder + "/ae_" +\
                               str(self.latent_dimension) +\
                               "_test" +\
                               str(self.id_topology) +\
                               ".ckpt"
        self.opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
        self.plot = plot
        self.metrics = ['accuracy']


        # Declaring the models
        self.encoder = Sequential()
        self.decoder = Sequential()
        self.autoencoder = Sequential()
        self.classifier = Sequential()
        self.model = Sequential()

        if self.id_topology == 1:
            # Setting Encoder architecture
            self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # 3x32x32 -> 32x16x16
            self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 32x16x16 -> 64x8x8
            self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 128x4x4
            self.encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            self.encoder.add(Flatten())
            self.encoder.add(Dense(self.latent_dimension, activation='relu'))

            # Setting Decoder architecture
            self.decoder.add(Dense(8*8*128, activation='relu'))
            self.decoder.add(Reshape((8, 8, 128)))
            self.decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            self.decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 128x4x4 -> 64x8x8
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 32x16x16
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # 32x16x16 -> 3x32x32

        if self.id_topology == 2:
            # Setting Encoder architecture
            self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # 3x32x32 -> 32x16x16
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 32x16x16 -> 64x8x8
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 128x4x4
            self.encoder.add(Flatten())
            self.encoder.add(Dense(self.latent_dimension, activation='relu'))

            # Setting Decoder architecture
            self.decoder.add(Dense(8*8*128, activation='relu'))
            self.decoder.add(Reshape((8, 8, 128)))
            self.decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 128x4x4 -> 64x8x8
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 32x16x16
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # 32x16x16 -> 3x32x32

        if self.id_topology == 3:
            # Setting Encoder architecture
            self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # 3x32x32 -> 32x16x16
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 32x16x16 -> 64x8x8
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Flatten())
            self.encoder.add(Dense(self.latent_dimension, activation='relu'))

            # Setting Decoder architecture
            self.decoder.add(Dense(8*8*64, activation='relu'))
            self.decoder.add(Reshape((8, 8, 64)))
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 32x16x16
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # 32x16x16 -> 3x32x32

        if self.id_topology == 4:
            # Setting Encoder architecture
            self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # 3x32x32 -> 32x16x16
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Flatten())
            self.encoder.add(Dense(self.latent_dimension, activation='relu'))

            # Setting Decoder architecture
            self.decoder.add(Dense(16*16*32, activation='relu'))
            self.decoder.add(Reshape((16, 16, 32)))
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # 32x16x16 -> 3x32x32

        if self.id_topology == 5:
            # Setting Encoder architecture
            self.encoder.add(Flatten())
            self.encoder.add(Dense(self.latent_dimension, activation='relu'))

            # Setting Decoder architecture
            self.decoder.add(Dense(32 * 32 * 3, activation='relu'))
            self.decoder.add(Reshape((32, 32, 3)))
            self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # 32x16x16 -> 3x32x32

        # This one is a variant of the topology 4, which uses no dense layers
        if self.id_topology == 6:
            # Setting Encoder architecture
            self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # 3x32x32 -> 32x16x16
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 32x16x16 -> 64x8x8
            self.encoder.add(MaxPool2D((2, 2), padding='same'))
            self.encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 128x4x4

            # Setting Decoder architecture
            self.decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 128x4x4 -> 64x8x8
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 32x16x16
            self.decoder.add(UpSampling2D((2, 2)))
            self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # 32x16x16 -> 3x32x32

        # Setting Autoencoder architecture
        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)

        # Setting Classifier architecture
        self.classifier.add(Flatten())
        self.classifier.add(Dense(64, activation='relu'))
        self.classifier.add(Dense(10, activation='softmax'))

        # Setting Model architecture
        self.model.add(self.encoder)
        self.model.add(self.classifier)

        # Compiling the model and the autoencoder
        self.model.compile(optimizer=self.opt, loss=loss, metrics=self.metrics)
        self.autoencoder.compile(optimizer=self.opt, loss=tf.keras.losses.MeanSquaredError(), metrics=self.metrics)

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

    def load_model(self):
        self.autoencoder.load_weights(self.checkpoint_path)

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
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.savefig('loss' + str(self.id_topology) + '.png')
            plt.show()

    def empiric_evaluation(self):
        (train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()

        train_images = train_images.astype('float64') / 255.
        test_images = test_images.astype('float64') / 255.

        encoded_imgs = self.encoder.predict(test_images)
        decoded_imgs = self.decoder.predict(encoded_imgs)

        for i in range(len(decoded_imgs)):
            n_samples = 10
            fig, axes = plt.subplots(2,n_samples)
            for j in range(n_samples):
                axes[0, j].imshow(test_images[i*n_samples+j].reshape(32, 32, 3))
                axes[1, j].imshow(decoded_imgs[i*n_samples+j].reshape(32, 32, 3))
            plt.show()



