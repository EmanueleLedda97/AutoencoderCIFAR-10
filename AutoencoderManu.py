import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class AutoencoderManu:

    def __init__(self, plot=True, loss='sparse_categorical_crossentropy', metrics=['accuracy']):

        # Setting dataset, checkpoint and other parameters
        self.dataset = keras.datasets.cifar10
        self.checkpoint_path = "cp_manu/cp.ckpt"
        self.opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-5)
        self.plot = plot

        # Setting model architecture
        self.model = keras.Sequential([
            keras.layers.Flatten(),                         # From image to flat vector
            keras.layers.Dense(128, activation='relu'),     # Introducing a 128-space hidden layer
            keras.layers.Dense(10, activation='softmax')    # Scoring each up of 10 labels
        ])

        # Compiling the model
        self.model.compile(optimizer=self.opt, loss=loss, metrics=metrics)

    def train(self):

        # Setting up the checkpoint
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        # Training the model
        (train_images, train_labels), (test_images, test_labels) = self.dataset.load_data()
        history = self.model.fit(train_images, train_labels,
                                 epochs=3,
                                 validation_data=(test_images, test_labels),
                                 callbacks=[cp_callback])

        # Plotting the results
        if self.plot:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.show()








