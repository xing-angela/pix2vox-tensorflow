import tensorflow as tf
import numpy as np

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

    def train(self, train_dataset, val_dataset):
        num_epochs = self.cfg.TRAIN.NUM_EPOCHES
        batch_size = self.cfg.CONST.BATCH_SIZE
        best_iou = -1
        try:
            for epoch in range(num_epochs):
                print('[INFO] %s Epoch [%d/%d].' %
                      (dt.now(), epoch, num_epochs))

                # Iterates through the batches
                self.train_batch(train_dataset, batch_size, epoch)

                # Validates the training model
                iou = self.test(val_dataset)

                if iou > best_iou:
                    best_iou = iou

        except KeyboardInterrupt as e:
            if epoch > 0:
                print(
                    "Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
            else:
                raise e

        # return stats

    def train_batch(self, train_dataset, batch_size, epoch_idx):
        imgs, vols = train_dataset[0], train_dataset[1]

        for batch_idx, end in enumerate(range(batch_size, len(imgs)+1, batch_size)):
            # Get the current batch of data
            start = end - batch_size
            batch_imgs = imgs[start:end]
            batch_vols = vols[start:end]

            # Perform a forward pass to train the encoder, decoder, merger, and refiner (if using)
            with tf.GradientTape() as tape:
                image_features = self.encoder(batch_imgs)
                raw_features, generated_volume = self.decoder(image_features)

                generated_volume = self.merger(
                    raw_features, generated_volume)

                encoder_loss = self.loss_function(
                    generated_volume, batch_vols) * 10

            # Update the weights based on the optimizer
            grads = tape.gradient(encoder_loss, self.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.trainable_variables))

            # Indicates the end of an epoch
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] EDLoss = %.4f'
                % (dt.now(), epoch_idx + 1, self.cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, len(imgs), encoder_loss))

    def test(self, dataset):
        num_epochs = self.cfg.TRAIN.NUM_EPOCHES
        batch_size = self.cfg.CONST.BATCH_SIZE

        for epoch in range(num_epochs):
            # Iterates through the batches
            test_iou = self.test_batch(dataset, batch_size, epoch)

            # Output testing results
            mean_iou = []
            for taxonomy_id in test_iou:
                test_iou[taxonomy_id]['iou'] = np.mean(
                    test_iou[taxonomy_id]['iou'], axis=0)
                mean_iou.append(test_iou[taxonomy_id]['iou']
                                * test_iou[taxonomy_id]['n_samples'])
            mean_iou = np.sum(mean_iou, axis=0) / len(dataset[0])

            max_iou = np.max(mean_iou)

            return max_iou

    def test_batch(self, dataset, batch_size, epoch_idx):
        imgs, vols = dataset[0], dataset[1]

        test_iou = dict()

        for batch_idx, end in enumerate(range(batch_size, len(imgs)+1, batch_size)):
            # Get the current batch of data
            start = end - batch_size
            batch_imgs = imgs[start:end]
            batch_vols = vols[start:end]

            # Perform a no-training forward pass to train the encoder, decoder, merger, and refiner (if using)
            image_features = self.encoder(batch_imgs)
            raw_features, generated_volume = self.decoder(image_features)

            generated_volume = self.merger(
                raw_features, generated_volume)

            encoder_loss = self.loss_function(
                generated_volume, batch_vols) * 10

            # IoU per sample
            sample_iou = []
            for th in self.cfg.TEST.VOXEL_THRESH:
                # check if the fucking float() shit works and the fucking item() shit
                _volume = tf.math.greater_equal(generated_volume, th).float()
                intersection = tf.math.reduce_sum(
                    tf.math.multiply(_volume, batch_vols)).float()
                union = tf.math.greater_equal(
                    tf.math.add(_volume, batch_vols), 1).float()
                # get rid of .item() shit
                sample_iou.append((intersection / union)[0])

            # IoU per taxonomy
            taxonomy_ids = dataset[2]
            taxonomy_id = taxonomy_ids[0] if isinstance(
                taxonomy_id[0], str) else taxonomy_id[0][0]
            sample_ids = dataset[3]
            sample_name = sample_ids[0]

            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f IoU = %s' %
                  (dt.now(), epoch_idx + 1, len(imgs), taxonomy_id, sample_name, encoder_loss, ['%.4f' % si for si in sample_iou]))

            return test_iou
