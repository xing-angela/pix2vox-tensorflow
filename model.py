import tensorflow as tf
import numpy as np
import os

from datetime import datetime as dt
from tensorboardX import SummaryWriter
import utils.network_utils
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger
from models.refiner import Refiner


class Pix2VoxModel(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)
        self.merger = Merger(self.cfg)
        self.refiner = None

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_function = loss

    ###################################### TESTING ######################################

    def train(self, train_dataset, val_dataset):
        '''
        Trains the model.

        :param train_dataset -- the training dataset which is a tuple of images, volumes, taxonomy names,
            and sample names
        :param val_dataset -- the evaluation dataset which is a tuple of images, volumes, taxonomy names, 
            and sample names
        '''

        # Set up the Tensorboard writer
        output_dir = os.path.join(
            self.cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
        log_dir = output_dir % 'logs'
        train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(log_dir, 'evaluation'))

        num_epochs = self.cfg.TRAIN.NUM_EPOCHES
        batch_size = self.cfg.CONST.BATCH_SIZE
        best_iou = -1

        # Training loop -- goes through each epoch
        for epoch in range(num_epochs):
            # Sets up the batch average metrics
            encoder_losses = utils.network_utils.AverageMeter()
            refiner_losses = utils.network_utils.AverageMeter()

            print('[INFO] %s Epoch [%d/%d].' % (dt.now(), epoch+1, num_epochs))

            # Iterates through the batches
            self.train_batch(train_dataset, batch_size, epoch,
                             train_writer, (encoder_losses, refiner_losses))

            # Append epoch loss to TensorBoard
            train_writer.add_scalar(
                'EncoderDecoder/EpochLoss', encoder_losses.avg, epoch + 1)
            train_writer.add_scalar(
                'Refiner/EpochLoss', refiner_losses.avg, epoch + 1)

            # Validates the training model
            iou = self.test(val_dataset, epoch, output_dir, val_writer)

            if iou > best_iou:
                best_iou = iou

        # Prints out the best iou for the training epoch
        print('[INFO] %s Best IoU = %f' % (dt.now(), best_iou))

        # Close SummaryWriter for TensorBoard
        train_writer.close()
        val_writer.close()

    def train_batch(self, train_dataset, batch_size, epoch_idx, writer, loss_meters):
        '''
        Trains a single batch.

        :param train_dataset -- the training dataset which is a tuple of images, volumes, taxonomy names,
            and sample names
        :param batch size -- the batch size
        :param epoch_idx -- the current epoch number (for logging purposes)
        :param writer -- the tensorboard SummaryWriter for training
        :param loss_meters -- the loss meters which is a tuple of the encoder loss meter and the refier 
            loss meter
        '''

        imgs, vols = train_dataset[0], train_dataset[1]
        encoder_losses, refiner_losses = loss_meters[0], loss_meters[1]

        num_batches = len(imgs) // batch_size

        for batch_idx, end in enumerate(range(batch_size, len(imgs)+1, batch_size)):
            # Get the current batch of data
            start = end - batch_size
            batch_imgs = imgs[start:end]
            batch_vols = vols[start:end]

            # Perform a forward pass to train the encoder, decoder, merger, and refiner (if using)
            with tf.GradientTape() as tape:
                image_features = self.encoder(batch_imgs, training=True)
                raw_features, generated_volumes = self.decoder(
                    image_features, training=True)

                generated_volumes = self.merger(
                    raw_features, generated_volumes, training=True)

                encoder_loss = self.loss_function(
                    batch_vols, generated_volumes) * 10

                # Uses the refiner if we are using Pix2Vox-A
                if self.cfg.TASK.MODEL_TYPE == 'A':
                    self.refiner = Refiner(self.cfg)
                    generated_volumes = self.refiner(
                        generated_volumes, training=True)

                    refiner_loss = self.loss_function(
                        batch_vols, generated_volumes) * 10
                    total_loss = encoder_loss + refiner_loss
                else:
                    refiner_loss = encoder_loss
                    total_loss = encoder_loss

            # Update the weights based on the optimizer
            grads = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.trainable_variables))

            # Indicates the end of an batch step
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] TotalLoss = %.4f EDLoss = %.4f RLoss = %.4f'
                % (dt.now(), epoch_idx + 1, self.cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, num_batches, total_loss, encoder_loss, refiner_loss))

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.numpy())
            refiner_losses.update(refiner_loss.numpy())

            # Append loss to TensorBoard
            n_itr = epoch_idx * num_batches + batch_idx
            writer.add_scalar(
                'EncoderDecoder/BatchLoss', encoder_loss.numpy(), n_itr)
            writer.add_scalar(
                'Refiner/BatchLoss', refiner_loss.numpy(), n_itr)

    ###################################### TESTING ######################################

    def test(self, dataset, epoch_idx, output_dir, writer=None):
        '''
        Tests the model.

        :param dataset -- the testing dataset which is a tuple of images, volumes, taxonomy names,
            and sample names
        :param epoch_idx -- the current epoch number (for logging purposes)
        :param output_dir -- the output directory for testing
        :param writer -- the tensorboard SummaryWriter, will default to None for testing and will 
            need input for evaluation
        '''

        encoder_losses = utils.network_utils.AverageMeter()
        refiner_losses = utils.network_utils.AverageMeter()

        # Iterates through the batches
        test_iou = self.test_batch(
            dataset, 1, epoch_idx, writer, (encoder_losses, refiner_losses), output_dir)

        # Output testing results
        mean_iou = []
        for taxonomy_id in test_iou:
            test_iou[taxonomy_id]['iou'] = np.mean(
                test_iou[taxonomy_id]['iou'], axis=0)
            mean_iou.append(test_iou[taxonomy_id]['iou']
                            * test_iou[taxonomy_id]['n_samples'])
        mean_iou = np.sum(mean_iou, axis=0) / len(dataset[0])
        max_iou = np.max(mean_iou)

        # Add testing results to TensorBoard
        if writer is not None:
            writer.add_scalar('EncoderDecoder/EpochLoss',
                              encoder_losses.avg, epoch_idx)
            writer.add_scalar('Refiner/EpochLoss',
                              refiner_losses.avg, epoch_idx)
            writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

        # Print the max iou
            print('[INFO] %s Max IoU = %f' % (dt.now(), max_iou))

        return max_iou

    def test_batch(self, dataset, batch_size, epoch_idx, writer, loss_meters, output_dir):
        '''
        Tests a single batch.

        :param dataset -- the testing dataset which is a tuple of images, volumes, taxonomy names,
            and sample names
        :param batch size -- the batch size
        :param epoch_idx -- the current epoch number (for logging purposes)
        :param writer -- the tensorboard SummaryWriter, will default to None for testing and will 
            need input for evaluation
        :param loss_meters -- the loss meters which is a tuple of the encoder loss meter and the refier 
            loss meter
        :param output_dir -- the output directory for testing
        '''
        imgs, vols = dataset[0], dataset[1]
        encoder_losses, refiner_losses = loss_meters[0], loss_meters[1]

        test_iou = dict()

        for batch_idx, end in enumerate(range(batch_size, len(imgs)+1, batch_size)):
            # Get the current batch of data
            start = end - batch_size
            batch_imgs = imgs[start:end]
            batch_vols = vols[start:end]

            # Perform a no-training forward pass to train the encoder, decoder, merger, and refiner (if using)
            image_features = self.encoder(batch_imgs, training=False)
            raw_features, generated_volume = self.decoder(
                image_features, training=False)

            generated_volume = self.merger(
                raw_features, generated_volume, training=False)

            encoder_loss = self.loss_function(
                generated_volume, batch_vols) * 10

            # Uses the refiner if we are using Pix2Vox-A
            if self.cfg.TASK.MODEL_TYPE == 'A':
                generated_volumes = self.refiner(
                    generated_volumes, training=False)
                refiner_loss = self.loss_function(
                    batch_vols, generated_volumes) * 10
            else:
                refiner_loss = encoder_loss

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.numpy())
            refiner_losses.update(refiner_loss.numpy())

            # IoU per sample
            sample_iou = []
            for th in self.cfg.TEST.VOXEL_THRESH:
                _volume = tf.cast(tf.math.greater_equal(
                    generated_volume, th), tf.float32)
                intersection = tf.math.reduce_sum(
                    tf.math.multiply(_volume, batch_vols)).numpy()
                union = tf.math.reduce_sum(tf.cast(tf.math.greater_equal(
                    tf.math.add(_volume, batch_vols), 1), tf.float32)).numpy()
                sample_iou.append((intersection / union))

            # IoU per taxonomy
            taxonomy_ids = dataset[2]
            taxonomy_id = taxonomy_ids[0] if isinstance(
                taxonomy_ids[0], str) else taxonomy_ids[0][0]
            sample_ids = dataset[3]
            sample_name = sample_ids[0]

            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Save the generated volumes
            if output_dir and batch_idx < 3:
                img_dir = output_dir % 'images'
                # Volume Visualization
                gv = generated_volume.cpu().numpy()
                utils.network_utils.save_volume(gv, os.path.join(
                    img_dir, 'test', 'epoch_%d' % epoch_idx), batch_idx)
                gtv = batch_vols
                utils.network_utils.save_volume(gtv, os.path.join(
                    img_dir, 'ground_truth', 'epoch_%d' % epoch_idx), batch_idx)

            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f IoU = %s' %
                  (dt.now(), batch_idx + 1, len(imgs) // batch_size, taxonomy_id, sample_name, encoder_loss, ['%.4f' % si for si in sample_iou]))

        return test_iou
