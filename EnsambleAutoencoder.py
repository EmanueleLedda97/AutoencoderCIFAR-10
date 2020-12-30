import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Softmax, Dropout, Reshape, BatchNormalization
from tensorflow.python.keras.layers import UpSampling2D
from PIL import Image


class EnsambleAutoencoder:

    def __init__(self, batch=128, lr=1e-5, label=1, latent=8, plot=True):

        # Setting dataset, checkpoint and other parameters
        self.batch_size = batch
        self.label_id = label
        self.latent_dimension = latent
        self.dataset = keras.datasets.cifar10
        self.checkpoint_path = "cp_ensamble/ae_latent" +str(self.latent_dimension) +"_label" +str(self.label_id) +".ckpt"
        self.opt = tf.keras.optimizers.Adam(lr=lr, decay=1e-5)
        self.plot = plot
        self.metrics = ['accuracy']

        # Declaring the models
        self.encoder = Sequential()
        self.decoder = Sequential()
        self.autoencoder = Sequential()
        self.classifier = Sequential()
        self.model = Sequential()

        # Setting Encoder architecture
        self.encoder.add(Conv2D(32, (3, 3),
                                activation='relu',
                                padding='same',
                                input_shape=(32, 32, 3)))  # 3x32x32 -> 32x32x32
        self.encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # 32x32x32 -> 32x32x32
        self.encoder.add(BatchNormalization())
        self.encoder.add(MaxPool2D((2, 2), padding='same'))  # 32x32x32 -> 32x16x16
        self.encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 32x16x16 -> 64x16x16
        self.encoder.add(BatchNormalization())
        self.encoder.add(MaxPool2D((2, 2), padding='same'))  # 64x16x16 -> 64x8x8
        self.encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 64x8x8 -> 128x8x8
        self.encoder.add(BatchNormalization())
        self.encoder.add(MaxPool2D((2, 2), padding='same'))  # 128x8x8 -> 128x4x4
        self.encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))  # 128x4x4 -> 256x4x4
        self.encoder.add(BatchNormalization())
        self.encoder.add(MaxPool2D((2, 2), padding='same'))  # 256x4x4 -> 256x2x2
        self.encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))  # 256x4x4 -> 512x2x2
        self.encoder.add(BatchNormalization())
        self.encoder.add(MaxPool2D((2, 2), padding='same'))  # 512x2x2 -> 512x1x1
        self.encoder.add(Flatten())  # 512x1x1 -> 512
        self.encoder.add(Dense(8, activation='relu'))  # 512 -> 8

        # Setting Decoder architecture
        self.decoder.add(Dense(512, activation='relu'))  # 8 -> 512
        self.decoder.add(Reshape((1, 1, 512)))  # 512 -> 512x1x1
        self.encoder.add(BatchNormalization())
        self.decoder.add(UpSampling2D((2, 2)))  # 512x1x1 -> 512x2x2
        self.decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))  # 512x2x2 -> 256x2x2
        self.encoder.add(BatchNormalization())
        self.decoder.add(UpSampling2D((2, 2)))  # 256x2x2 -> 256x4x4
        self.decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 256x4x4 -> 128x4x4
        self.encoder.add(BatchNormalization())
        self.decoder.add(UpSampling2D((2, 2)))  # 128x4x4 -> 128x8x8
        self.decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 128x8x8 -> 64x8x8
        self.encoder.add(BatchNormalization())
        self.decoder.add(UpSampling2D((2, 2)))  # 64x8x8 -> 64x16x16
        self.decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # 64x16x16 -> 32x16x16
        self.encoder.add(BatchNormalization())
        self.decoder.add(UpSampling2D((2, 2)))  # 32x16x16 -> 32x32x32
        self.decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # 32x32x32 -> 3x32x32

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
        self.model.compile(optimizer=self.opt, loss='sparse_categorical_crossentropy', metrics=self.metrics)
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
        mask_test = np.reshape((test_labels == self.label_id), newshape=(10000,))
        mask_train = np.reshape((train_labels == self.label_id), newshape=(50000,))
        test_images = test_images[mask_test]
        train_images = train_images[mask_train]
        history = self.autoencoder.fit(train_images, train_images,
                                       epochs=epochs,
                                       validation_data=(test_images, test_images),
                                       callbacks=[cp_callback])

        # Plotting the results
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.savefig('cp_ensamble/losses/loss' + str(self.label_id) + '_latent' + str(self.latent_dimension) + '.png')
        if self.plot:
            plt.show()
        plt.clf()

    def evaluate(self):
        (_, _), (test_images, test_labels) = self.dataset.load_data()
        mask_test = np.reshape((test_labels == self.label_id), newshape=(10000,))
        test_images = test_images[mask_test]
        test_images = test_images.astype('float32') / 255.
        return self.autoencoder.evaluate(test_images, test_images)

    def empiric_evaluation(self):
        (train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()

        mask_test = np.reshape((test_labels == self.label_id), newshape=(10000,))
        mask_train = np.reshape((train_labels == self.label_id), newshape=(50000,))
        test_images = test_images[mask_test]
        train_images = train_images[mask_train]

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










