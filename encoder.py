import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16 = tf.keras.applications.vgg16.VGG16(weights = 'imagenet')
        self.vgg = tf.keras.models.Sequential([*list(vgg16.layers)]) # want to translate from pytorch: .features.children()
        self.layer1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2d(filters=512, kernel_size=3, activation='relu'), # pytorch: in_channels=512, out_channels=512, activation is not a parameter
            tf.keras.layers.BarchNormalization(),
            tf.keras.layers.ELU() #change to Dense with activation 'elu'?
        ])
        self.layer2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2d(filters=512, kernel_size=3, activation='relu'),
            tf.keras.layers.BarchNormalization(),
            tf.keras.layers.ELU(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3)) # kernel_size = 3 
        ])
        self.layer3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2d(filters=256, kernel_size=1, activation='relu'), # pytorch: in_channels=512, out_channels=256
            tf.keras.layers.BarchNormalization(),
            tf.keras.layers.ELU(),
        ])


        '''
        I think that this part is unnecessary in Tensorflow, but could be wrong

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False
        '''

    def forward(self, rendering_images):
        rendering_images = tf.reshape(rendering_images, perm = [1, 0, 2, 3, 4])
        rendering_images = tf.split(rendering_images, 1, axis=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(tf.squeeze(input=img, axis=0))
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            image_features.append(features)

        image_features = tf.reshape(tf.stack(image_features), perm = [1, 0, 2, 3, 4])

        return image_features


