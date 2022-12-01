import tensorflow as tf

from model.encoder import Encoder
from model.decoder import Decoder
from model.merger import Merger
from model.refiner import Refiner


class Pix2VoxModel(tf.keras.Model):
    def __init__(self, augment_fn=lambda x: x, use_refiner=False, **kwargs):
        super().__init__(**kwargs)
        """
        Initializes the custom pix2vox model.

        :param augment_fn: the augmentation function used for augmenting images
        :param use_refiner: whether to use the refiner -- True if we are using Pix2VoxA and 
                            False if we are using Pix2VoxF
        """
        self.augment_fn = augment_fn
        self.use_refiner = use_refiner

        # the layers
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.merger = Merger()
        self.refiner = None

        if self.use_refiner:
            self.refiner = Refiner()

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss
        self.accuracy_function = metrics[0]

    def train(self, num_epochs, train_dataset_loader, val_dataset_loader):
        """
        The training loop for the model.

        :param num_epochs: the number of epochs
        :param train_dataset_loader: the custom dataset loader for the chosen dataset
        :param val_dataset_loader: the custom dataset loader for the chosen dataset
        """

        for e in range(num_epochs):
            # iterates through each batch
            for b, datum in enumerate(train_dataset_loader.data):
                # gets the data from the data loader
                taxonomy_names, split_names = datum[0], datum[1]
                images, gt_vols = datum[2], datum[3]

                with tf.GradientTape() as tape:
                    # sends the images through the encoder and decoder
                    features = self.encoder(images)
                    raw_features, generated_vols = self.decoder(features)

                    # sends the extracted features and volumes through the merger
                    generated_vols = merger(raw_features, generated_vols)

                    # calculates the loss using binary crossentropy
                    bce_loss = tf.keras.losses.BinaryCrossentropy()
                    loss = bce_loss(gt_vols, generated_vols)

                # computes the gradients
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.trainable_variables))
