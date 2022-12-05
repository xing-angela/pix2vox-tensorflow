import argparse
import utils.data_loader as data_loader
import tensorflow as tf
from pix2vox_model import Pix2VoxModel
from datetime import datetime as dt


def parse_args():
    """
    Parses the command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Parse Data for Pix2Vox")

    parser.add_argument("--dataset", default="ShapeNet",
                        choices=["ShapeNet"], help="Dataset to use")
    parser.add_argument("--task", default="train",
                        choices=["train, test", "both"], help="Type of task (train, test, both)")
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
    loader = data_loader.DATASET_MAPPINGS[args.dataset](args)
    data = []

    if args.task == "train":
        # gets the training data
        dataset_files = loader.load_dataset_files("train")
        data = loader.load_data("train", dataset_files, 1)

    elif args.task == "val":
        # gets the evaluation data
        dataset_files = loader.load_dataset_files("val")
        data = loader.load_data("val", dataset_files, 1)

    elif args.task == "test":
        # gets the testing data
        dataset_files = loader.load_dataset_files("test")
        data = loader.load_data("test", dataset_files, 1)

    return data


def main():
    """
    Parses and preprocesses data and runs the model.
    """

    args = parse_args()

    X0, Y0, X1, Y1 = [], [], [], []

    ######################## training the data ########################
    # parses the data depnding on training or testing
    if args.task in ["train", "both"]:
        train_data_loader = get_data(args)

    # augment_prep_fn = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.RandomCrop(width=224, height=224),
    #         tf.keras.layers.RandomFlip(),
    #         tf.keras.layers.RandomRotation(factor=0.05)
    #     ]
    # )

    # sets up the model
    model = Pix2VoxModel(args)

    compile_model(model, args)

    print('[INFO] %s Starting training for %d epochs with a batch size of %d.' % (
        dt.now(), 250, 2))
    model.train(250, 2, train_data_loader)


def compile_model(model, args):
    '''Compiles model by reference based on arguments'''
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy_function"]  # CHANGE THIS
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
