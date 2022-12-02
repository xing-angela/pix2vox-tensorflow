import tensorflow as tf

from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.merger import Merger
# from model.refiner import Refiner


class Pix2VoxModel(tf.keras.Model):
    def __init__(self, args, augment_fn=lambda x: x, use_refiner=False, **kwargs):
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
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.merger = Merger(args)
        self.refiner = None

        # if self.use_refiner:
        #     self.refiner = Refiner()

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss
        self.accuracy_function = metrics[0]

    def train(self, num_epochs, batch_size, dataset):
        """
        The training loop for the model.

        :param num_epochs: the number of epochs
        :param batch_size: the batch size
        :param dataset: the dataset (containing images and volumes)
        """

        for e in range(num_epochs):
            # iterates through each batch
            for batch_idx, end in enumerate(range(batch_size, len(dataset)+1, batch_size)):
                # gets the current batch of data
                start = end - batch_size
                batch_datums = dataset[start:end]

                # gets the data from the batch
                images, gt_vols = [], []

                for datum in batch_datums:
                    images.append(datum[2])
                    gt_vols.append(datum[3])

                images = tf.constant(images)
                gt_vols = tf.constant(gt_vols)

                # augments the images for training
                # images = self.augment_fn(images)

                with tf.GradientTape() as tape:
                    # sends the images through the encoder and decoder
                    features = self.encoder(images)
                    raw_features, generated_vols = self.decoder(features)

                    # sends the extracted features and volumes through the merger
                    generated_vols = self.merger(raw_features, generated_vols)

                    # calculates the loss
                    loss = self.loss_function(gt_vols, generated_vols)

                # computes the gradients
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.trainable_variables))
