import argparse
import utils.data_loader as data_loader
import tensorflow as tf


def parse_args():
    """
    Parses the command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Parse Data for Pix2Vox")

    parser.add_argument("--dataset", required=True,
                        choices=["ShapeNet"], help="Dataset to use")
    parser.add_argument("--task", default="train", required=True,
                        choices=["train, test"], help="Type of task (train, test, both)")
    parser.add_argument('--epoch', help='Number of epoches',
                        default=250, type=int)
    parser.add_argument('--out', default="output", help='Set output path')
    parser.add_argument('--type', default="F",
                        choices=["A", "F"], help="Type of model -- Pix2VoxA or Pix2VoxF")

    return parser.parse_args()


def get_data(args):
    """
    Sets up the data loader and return the data based on whether it's training or testing.

    :param args: the arguments from the command line
    """
    # sets up the data loader
    train_data_loader = data_loader.DATASET_MAPPINGS[args.dataset](args)

    x0, y0, x1, y1 = [], [], [], []

    if args.task == "train":
        # gets the training data
        train_dataset = train_data_loader.load_dataset(
            data_loader.DatasetType.TRAIN, 1)
        train_dataset.set_data()
        x0, y0 = train_dataset.images, train_dataset.volumes

        # gets the eval data
        val_dataset = train_data_loader.load_dataset(
            data_loader.DatasetType.VAL, 1)
        val_dataset.set_data()
        x1, y1 = val_dataset.images, val_dataset.volumes

    return (x0, y0, x1, y1)


def main():
    """
    Parses and preprocesses data and runs the model.
    """

    args = parse_args()

    X0, Y0, X1, Y1 = [], [], [], []

    # parses the data depnding on training or testing
    if args.task == "train":
        X0, Y0, X1, Y1 = get_data(args)

    augment_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.RandomCrop(width=224, height=224),
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(factor=0.05)
        ]
    )


if __name__ == "__main__":
    main()


######################## for viewing the images ########################
# rand_indices = random.sample(range(0, len(images)), 10)
# rand_imgs = [images[i] for i in rand_indices]
# augmented = augment_prep_fn(rand_imgs)

# fig = plt.figure()

# for i in range(1, len(rand_imgs)+1):
#     fig.add_subplot(5, 5, i)
#     plt.imshow(augmented[i-1])
# plt.show()

# fig, ax = plt.subplots(2, 10)
# for i in range(1, len(rand_imgs)+1):
#     ax[1][i-1].imshow(images[rand_indices[i-1]])
#     ax[0][i-1].imshow(augmented[i-1])
# plt.show()
