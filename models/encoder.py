import tensorflow as tf
import numpy as np


class Encoder(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        # Layer Definition
        vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet')
        self.vgg = tf.keras.models.Sequential(vgg16.layers[:12])
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=512, kernel_size=3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ELU()
        ])
        self.layer2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=512, kernel_size=3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ELU(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3))
        ])
        self.layer3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ELU()
        ])

    def call(self, rendering_images):
        rendering_images = np.transpose(rendering_images, [1, 0, 2, 3, 4])
        rendering_images = np.split(rendering_images, 1, axis=0)
        image_features = []

        for img in rendering_images:
            formatted = tf.keras.applications.vgg16.preprocess_input(
                np.squeeze(img, axis=0))
            features = self.vgg(formatted)
            # print('vgg shape: ', features.shape) # [batch size, 28, 28, 512]
            features = self.layer1(features)
            # print('layer1 shape: ', features.shape) # [batch size, 26, 26, 512]
            features = self.layer2(features)
            # print('layer2 shape: ', features.shape) # [batch size, 24, 24, 512]
            features = self.layer3(features)
            # print('layer shape: ', features.shape) # [batch size, 28, 28, 512]
            image_features.append(features)

        image_features = tf.transpose(
            tf.stack(image_features), perm=[1, 0, 2, 3, 4])
        # print("output feature shape: ", image_features.shape) # [batch size, n_views, 28, 28, 256]

        return image_features
