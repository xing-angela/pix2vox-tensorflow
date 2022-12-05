import tensorflow as tf

from datetime import datetime as dt
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger


class Pix2VoxModel(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)
        self.merger = Merger(self.cfg)

    def compile(self, optimizer, loss):  # , metrics
        self.optimizer = optimizer
        self.loss_function = loss
        # self.accuracy_function = metrics[0]

    def train(self, dataset):
        num_epochs = self.cfg.TRAIN.NUM_EPOCHES
        batch_size = self.cfg.CONST.BATCH_SIZE
        stats = []
        try:
            for epoch in range(num_epochs):
                print('[INFO] %s Epoch [%d/%d].' %
                      (dt.now(), epoch, num_epochs))
                stats += [self.train_batch(dataset, batch_size, epoch)]
        except KeyboardInterrupt as e:
            if epoch > 0:
                print(
                    "Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
            else:
                raise e

        return stats

    def train_batch(self, dataset, batch_size, epoch_idx):
        imgs, vols = dataset[0], dataset[1]

        for batch_idx, end in enumerate(range(batch_size, len(imgs)+1, batch_size)):
            # Get the current batch of data
            start = end - batch_size
            batch_imgs = imgs[start:end]
            batch_vols = vols[start:end]

            # Perform a forward pass to train the encoder, decoder, merger, and refiner (if using)
            with tf.GradientTape() as tape:
                image_features = self.encoder(batch_imgs)
                raw_features, generated_volumes = self.decoder(image_features)

                generated_volumes = self.merger(
                    raw_features, generated_volumes)

                encoder_loss = self.loss_function(
                    generated_volumes, batch_vols) * 10

            # update the weights based on the optimizer
            grads = tape.gradient(encoder_loss, self.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.trainable_variables))

            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] EDLoss = %.4f'
                % (dt.now(), epoch_idx + 1, self.cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, len(imgs), encoder_loss))
