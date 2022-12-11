import tensorflow as tf


class Refiner(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        dense_kernel_init = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.01)
        dense_bias_init = tf.keras.initializers.Constant(value=0)

        self.layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv3D(
                filters=32, kernel_size=2, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE),
            tf.keras.layers.MaxPool3D(pool_size=2, strides=2, trainable=False)
        ])

        self.layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv3D(
                filters=64, kernel_size=2, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE),
            tf.keras.layers.MaxPool3D(pool_size=2, strides=2, trainable=False)
        ])

        self.layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv3D(
                filters=128, kernel_size=2, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(self.cfg.NETWORK.LEAKY_VALUE),
            tf.keras.layers.MaxPool3D(pool_size=2, strides=2, trainable=False)
        ])

        self.layer4 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                2048, activation='relu', kernel_initializer=dense_kernel_init, bias_initializer=dense_bias_init)
        ])

        self.layer5 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                8192, activation='relu', kernel_initializer=dense_kernel_init, bias_initializer=dense_bias_init),
        ])

        self.layer6 = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(
                64, kernel_size=4, strides=2, use_bias=self.cfg.NETWORK.TCONV_USE_BIAS, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.layer7 = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(
                32, kernel_size=4, strides=2, use_bias=self.cfg.NETWORK.TCONV_USE_BIAS, padding='same', kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        self.layer8 = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(
                1, kernel_size=4, strides=2, use_bias=self.cfg.NETWORK.TCONV_USE_BIAS, padding='same', kernel_initializer='he_normal', activation='sigmoid')
        ])

    def call(self, coarse_volumes):
        volumes_32_l = tf.reshape(
            coarse_volumes, (-1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, 1))
        # print("volumes_32_l shape: ", volumes_32_l.shape) # (batch_size, 32, 32, 32, 1)
        volumes_16_l = self.layer1(volumes_32_l)
        # print("volumes_16_l shape: ", volumes_16_l.shape) # (batch_size, 16, 16, 16, 32)
        volumes_8_l = self.layer2(volumes_16_l)
        # print("volumes_8_l shape: ", volumes_8_l.shape) # (batch_size, 8, 8, 8, 64)
        volumes_4_l = self.layer3(volumes_8_l)
        # print("volumes_4_l shape: ", volumes_4_l.shape) # (batch_size, 4, 4, 4, 128)
        flatten_features = self.layer4(tf.reshape(volumes_4_l, (-1, 8192)))
        # print("flatten_features 1 shape: ", flatten_features.shape) # (batch_size, 2048)
        flatten_features = self.layer5(flatten_features)
        # print("flatten_features 2 shape: ", flatten_features.shape) # (batch_size, 8192)
        volumes_4_r = volumes_4_l + \
            tf.reshape(flatten_features, (-1, 4, 4, 4, 128))
        # print("volumes_4_r shape: ", volumes_4_r.shape) # (batch_size, 4, 4, 4, 128)
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print("volumes_8_r shape: ", volumes_8_r.shape) # (batch_size, 8, 8, 8, 64)
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print("volumes_16_r shape: ", volumes_16_r.shape) # (batch_size, 16, 16, 16, 32)
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print("volumes_32_r shape: ", volumes_32_r.shape) # (batch_size, 32, 32, 32, 1)

        return tf.reshape(volumes_32_r, (-1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))
