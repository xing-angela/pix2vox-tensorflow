import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        # Layer Definition
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=512, kernel_size=4, strides=2,
                                            use_bias=cfg.NETWORK.TCONV_USE_BIAS, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.layer2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=128, kernel_size=4, strides=2,
                                            use_bias=cfg.NETWORK.TCONV_USE_BIAS, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.layer3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=4, strides=2,
                                            use_bias=cfg.NETWORK.TCONV_USE_BIAS, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.layer4 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=8, kernel_size=4, strides=2,
                                            use_bias=cfg.NETWORK.TCONV_USE_BIAS, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.layer5 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=1, strides=1,
                                            use_bias=cfg.NETWORK.TCONV_USE_BIAS, padding='valid', activation='sigmoid'),
        ])

    def call(self, image_features):
        image_features = tf.transpose(image_features, perm=[1, 0, 2, 3, 4])
        image_features = tf.split(image_features, 1, axis=0)
        gen_volumes = []
        raw_features = []

        for features in image_features:
            gen_volume = tf.reshape(features, (-1, 2, 2, 2, 2048))
            # print("reshape shape: ", gen_volume.shape) # (batch_size, 2, 2, 2, 2048)
            gen_volume = self.layer1(gen_volume)
            # print("layer 1 shape: ", gen_volume.shape) # (batch_size, 4, 4, 4, 512)
            gen_volume = self.layer2(gen_volume)
            # print("layer 2 shape: ", gen_volume.shape) # (batch_size, 8, 8, 8, 128)
            gen_volume = self.layer3(gen_volume)
            # print("layer 3 shape: ", gen_volume.shape) # (batch_size, 16, 16, 16, 32)
            gen_volume = self.layer4(gen_volume)
            # print("layer 4 shape: ", gen_volume.shape) # (batch_size, 32, 32, 32, 8)
            raw_feature = gen_volume
            gen_volume = self.layer5(gen_volume)
            # print("layer 5 shape: ", gen_volume.shape) # (batch_size, 32, 32, 32, 1)
            raw_feature = tf.concat((raw_feature, gen_volume), axis=4)
            # print("raw feature shape: ", raw_feature.shape) # (batch_size, 32, 32, 32, 9)

            gen_volumes.append(tf.squeeze(gen_volume, axis=4))
            raw_features.append(raw_feature)

        gen_volumes = tf.transpose(tf.stack(gen_volumes), perm=[1, 0, 2, 3, 4])
        # print("gen volume shape: ", gen_volumes.shape) # (batch_size, n_views, 32, 32, 32])
        raw_features = tf.transpose(
            tf.stack(raw_features), perm=[1, 0, 2, 3, 4, 5])
        # print("raw features shape: ", raw_features.shape) # (batch_size, n_views, 32, 32, 32, 9)

        return raw_features, gen_volumes
