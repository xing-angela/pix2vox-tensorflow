import tensorflow as tf

class Refiner(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        
        self.layer1 = tf.keras.Sequential()
        self.layer1.add(tf.keras.layers.Conv3D(1, 32, kernel_size=4, padding=2))
        self.layer1.add(tf.keras.layers.BatchNormalization())
        self.layer1.add(tf.keras.layers.LeakyReLU(0.2))
        self.layer1.add(tf.keras.layers.MaxPool3D(pool_size=2))

        self.layer2 = tf.keras.Sequential()
        self.layer2.add(tf.keras.layers.Conv3D(1, 32, kernel_size=4, padding=2))
        self.layer2.add(tf.keras.layers.BatchNormalization())
        self.layer2.add(tf.keras.layers.LeakyReLU(0.2))
        self.layer2.add(tf.keras.layers.MaxPool3D(pool_size=2))

        self.layer3 = tf.keras.Sequential()
        self.layer3.add(tf.keras.layers.Conv3D(64, 128, kernel_size=4, padding=2))
        self.layer3.add(tf.keras.layers.BatchNormalization())
        self.layer3.add(tf.keras.layers.LeakyReLU(0.2))
        self.layer3.add(tf.keras.layers.MaxPool3D(pool_size=2))

        self.layer4 = tf.keras.Sequential()
        self.layer4.add(tf.keras.layers.Dense(2048, input_shape=(8192,), activation=None))
        self.layer4.add(tf.keras.layers.ReLU())

        self.layer5 = tf.keras.Sequential()
        self.layer5.add(tf.keras.layers.Dense(8192, input_shape=(2048,), activation=None))
        self.layer5.add(tf.keras.layers.ReLU())

        self.layer6 = tf.keras.Sequential()
        self.layer6.add(tf.keras.layers.Conv3DTranspose(128, 64, kernel_size=4, stride=2, use_bias=False, padding=1))
        self.layer6.add(tf.keras.layers.BatchNormalization())
        self.layer6.add(tf.keras.layers.ReLU())

        self.layer7 = tf.keras.Sequential()
        self.layer7.add(tf.keras.layers.Conv3DTranspose(64, 32, kernel_size=4, stride=2, use_bias=False, padding=1))
        self.layer7.add(tf.keras.layers.BatchNormalization())
        self.layer7.add(tf.keras.layers.ReLU())

    def call(self, coarse_volumes):
        volumes_32_l = coarse_volumes.view((-1, 1,32, 32, 32))
        volumes_16_l = self.layer1(volumes_32_l)
        volumes_8_l = self.layer2(volumes_16_l)
        volumes_4_l = self.layer3(volumes_8_l)
        flatten_features = self.layer4(volumes_4_l.view(-1, 8192))
        flatten_features = self.layer5(flatten_features)
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5

        return volumes_32_r.view((-1,32,32,32))
#