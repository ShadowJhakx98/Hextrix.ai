"""
toy_text_to_image.py

A very simplified "GAN" that outputs 28x28 images from
a random latent vector. We don't actually do real text->image
mapping; it's purely for demonstration.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten

class ToyTextToImage:
    def __init__(self):
        """
        Build a simple generator + discriminator. For
        demonstration, we won't do a full training step here.
        """
        self.generator, self.discriminator, self.gan_model = self._build_gan()

    def _build_gan(self):
        # Generator
        z = layers.Input(shape=(100,))
        g = Dense(128, activation='relu')(z)
        g = Dense(28*28, activation='sigmoid')(g)
        img = Reshape((28,28,1))(g)
        generator = Model(z, img)

        # Discriminator
        i = layers.Input(shape=(28,28,1))
        d = Flatten()(i)
        d = Dense(128, activation='relu')(d)
        d = Dense(1, activation='sigmoid')(d)
        discriminator = Model(i, d)
        discriminator.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
        discriminator.trainable = False

        # Combined model
        gan_input = layers.Input(shape=(100,))
        gen_out = generator(gan_input)
        gan_out = discriminator(gen_out)
        gan_model = Model(gan_input, gan_out)
        gan_model.compile(optimizer='adam', loss='binary_crossentropy')

        return generator, discriminator, gan_model

    def text_to_image(self, prompt: str):
        """
        For now, we just create a random latent z. In a real approach,
        you'd embed the prompt and incorporate it into the generator input.
        """
        z = np.random.normal(0,1,size=(1,100))
        # Possibly parse the prompt -> embed -> transform -> z, etc.
        gen_img = self.generator.predict(z)
        return gen_img  # shape (1,28,28,1)
