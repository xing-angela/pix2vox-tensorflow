import tensorflow as tf

class Merger(tf.keras.layers.Layer):

    def __init__(self, cfg):
        super(Merger, self).__init__()

        self.layer1 = tf.keras.Sequential()
        self.layer1.add(tf.keras.layers.Conv3D(9, 16, kernel_size=3, padding=1))
        self.layer1.add(tf.keras.layers.BatchNormalization())
        self.layer1.add(tf.keras.layers.LeakyReLU(0.2))

        self.layer2 = tf.keras.Sequential()
        self.layer2.add(tf.keras.layers.Conv3D(16, 8, kernel_size=3, padding=1))
        self.layer2.add(tf.keras.layers.BatchNormalization())
        self.layer2.add(tf.keras.layers.LeakyReLU(0.2))

        self.layer3 = tf.keras.Sequential()
        self.layer3.add(tf.keras.layers.Conv3D(8,4, kernel_size=3, padding=1))
        self.layer3.add(tf.keras.layers.BatchNormalization())
        self.layer3.add(tf.keras.layers.LeakyReLU(0.2))

        self.layer4 = tf.keras.Sequential()
        self.layer4.add(tf.keras.layers.Conv3D(4,2, kernel_size=3, padding=1))
        self.layer4.add(tf.keras.layers.BatchNormalization())
        self.layer4.add(tf.keras.layers.LeakyReLU(0.2))

        self.layer5 = tf.keras.Sequential()
        self.layer5.add(tf.keras.layers.Conv3D(4,2, kernel_size=3, padding=1))
        self.layer5.add(tf.keras.layers.BatchNormalization())
        self.layer5.add(tf.keras.layers.LeakyReLU(0.2))

    
    def call(self, raw_features, coarse_volumes):
        n_views_rendering = coarse_volumes.size(1)
        tf.split(raw_features, 1, dim=1)
        volume_weights = []

        for i in range(n_views_rendering):
            raw_feature = tf.squeze(raw_features[i], dim=1)

            volume_weight = self.layer1(raw_feature)
            volume_weight = self.layer2(volume_weight)
            volume_weight = self.layer3(volume_weight)
            volume_weight = self.layer4(volume_weight)
            volume_weight = self.layer5(volume_weight)

            volume_weight = tf.squeeze(volume_weight, dim=1)
            volume_weights.append(volume_weight)
        
        volume_weights = tf.stack(volume_weights).transpose(volume_weights,(1, 0, 2, 3, 4))
        volume_weights = tf.keras.activations.softmax(volume_weights)
        coarse_volumes = coarse_volumes * volume_weights
        coarse_volumes = tf.math.reduce_sum(coarse_volumes, axis=1)
        
        return tf.clip_by_value(coarse_volumes, clip_value_min=0, clip_value_max=1)
        #