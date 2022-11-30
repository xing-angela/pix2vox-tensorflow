import argparse
import utils.data_loader as data_loader
import matplotlib.pyplot as plt
import random
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse Data for Pix2Vox")

    parser.add_argument("--dataset", help="the dataset to use")

    return parser.parse_args()


def main():
    args = parse_args()

    # sets up the data loader
    train_data_loader = data_loader.DATASET_MAPPINGS["ShapeNet"](args)
    dataset = train_data_loader.load_dataset(data_loader.DatasetType.TRAIN, 1)
    dataset.set_data()

    images = dataset.images

    augment_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.RandomCrop(width=224, height=224),
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(factor=0.05)
        ]
    )

    rand_indices = random.sample(range(0, len(images)), 10)
    rand_imgs = [images[i] for i in rand_indices]
    augmented = augment_prep_fn(rand_imgs)

    # fig = plt.figure()

    # for i in range(1, len(rand_imgs)+1):
    #     fig.add_subplot(5, 5, i)
    #     plt.imshow(augmented[i-1])
    # plt.show()

    fig, ax = plt.subplots(2, 10)
    for i in range(1, len(rand_imgs)+1):
        ax[1][i-1].imshow(images[rand_indices[i-1]])
        ax[0][i-1].imshow(augmented[i-1])
    plt.show()


if __name__ == "__main__":
    main()
