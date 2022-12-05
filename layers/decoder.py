import tensorflow as tf


class Decoder(tf.keras.layers.Layer):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args

        # Layer Definition
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(
                filters=512, kernel_size=4, strides=2, use_bias=True, padding='same', activation='relu'),
            # pytorch: in_channels=2048, out_channels=512, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1, activation is not a parameter
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.layer2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=128, kernel_size=4, strides=2,
                                            use_bias=True, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.layer3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=4, strides=2,
                                            use_bias=True, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.layer4 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=8, kernel_size=4, strides=2,
                                            use_bias=True, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.layer5 = tf.keras.models.Sequential([
            tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=1, strides=1,
                                            use_bias=True, padding='valid', activation='sigmoid'),
        ])

    def call(self, image_features):
        image_features = tf.transpose(image_features, perm=[1, 0, 2, 3, 4])
        image_features = tf.split(image_features, 1, axis=0)
        gen_volumes = []
        raw_features = []

        for features in image_features:
            gen_volume = tf.reshape(features, (-1, 2048, 2, 2, 2))
            print("reshape shape (batch_size, 2048, 2, 2, 2): ", gen_volume.shape)
            gen_volume = self.layer1(gen_volume)
            print("layer 1 shape (batch_size, 512, 4, 4, 4): ", gen_volume.shape)
            gen_volume = self.layer2(gen_volume)
            print("layer 2 shape (batch_size, 128, 8, 8, 8): ", gen_volume.shape)
            gen_volume = self.layer3(gen_volume)
            print("layer 3 shape (batch_size, 32, 16, 16, 16): ", gen_volume.shape)
            gen_volume = self.layer4(gen_volume)
            print("layer 4 shape (batch_size, 32, 16, 16, 16): ", gen_volume.shape)
            raw_feature = gen_volume
            fuck
            gen_volume = self.layer5(gen_volume)
            raw_feature = tf.concat((raw_feature, gen_volume), axis=1)

            gen_volumes.append(tf.squeeze(gen_volume, axis=1))
            raw_features.append(raw_feature)

        gen_volumes = tf.transpose(tf.stack(gen_volumes), perm=[1, 0, 2, 3, 4])
        raw_features = tf.transpose(
            tf.stack(raw_features), perm=[1, 0, 2, 3, 4])

        return raw_features, gen_volumes
