import tensorflow as tf


class Merger(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        # Layer Definition
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3D(
                filters=16, kernel_size=3, padding='same',
                kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
        ])
        self.layer2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3D(
                filters=8, kernel_size=3, padding='same',
                kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
        ])
        self.layer3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3D(
                filters=4, kernel_size=3, padding='same',
                kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
        ])
        self.layer4 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3D(
                filters=2, kernel_size=3, padding='same',
                kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
        ])
        self.layer5 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3D(
                filters=1, kernel_size=3, padding='same',
                kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
        ])

    def call(self, raw_features, coarse_volumes):
        n_views_rendering = coarse_volumes.shape[1]
        raw_features = tf.split(raw_features, 1, axis=1)
        volume_weights = []

        for i in range(n_views_rendering):
            raw_feature = tf.squeeze(raw_features[i], axis=1)
            # print('raw features shape: ', raw_feature.shape) # [batch_size, 32, 32, 32, 9]

            volume_weight = self.layer1(raw_feature)
            # print('layer 1 shape: ', volume_weight.shape) # [batch_size, 32, 32, 32, 16]
            volume_weight = self.layer2(volume_weight)
            # print('layer 2 shape: ', volume_weight.shape) # [batch_size, 32, 32, 32, 8]
            volume_weight = self.layer3(volume_weight)
            # print('layer 3 shape: ', volume_weight.shape) # [batch_size, 32, 32, 32, 4]
            volume_weight = self.layer4(volume_weight)
            # print('layer 4 shape: ', volume_weight.shape) # [batch_size, 32, 32, 32, 2]
            volume_weight = self.layer5(volume_weight)
            # print('layer 5 shape: ', volume_weight.shape) # [batch_size, 32, 32, 32, 1]

            volume_weight = tf.squeeze(volume_weight, axis=4)
            volume_weights.append(volume_weight)

        volume_weights = tf.transpose(
            tf.stack(volume_weights), (1, 0, 2, 3, 4))
        volume_weights = tf.nn.softmax(volume_weights)
        # print("volume weights shape: ", volume_weights.shape) # [batch_size, n_views, 32, 32, 32]
        coarse_volumes = coarse_volumes * volume_weights
        coarse_volumes = tf.math.reduce_sum(coarse_volumes, axis=1)
        # print("coarse volumes shape: ", coarse_volumes.shape) # [batch_size, n_views, 32, 32, 32]

        return tf.clip_by_value(coarse_volumes, clip_value_min=0, clip_value_max=1)
