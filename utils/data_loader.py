import json
import os
import random
import cv2
import numpy as np
import sys
from datetime import datetime as dt
from enum import Enum
import sys
import utils.binvox_rw


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2

###################################### SHAPENET ######################################


class ShapeNetDataset:
    def __init__(self, dataset_type, data_files, n_views_rendering):
        self.dataset_type = dataset_type
        self.data_files = data_files
        self.n_views_rendering = n_views_rendering
        self.data = []

    def set_data(self):
        for i in range(len(self.data_files)):
            datum = self.get_datum(i)
            self.data.extend(datum)

        print('[INFO] %s Complete collecting saving images and volumes of dataset.' % (
            dt.now()))

    def get_datum(self, idx):
        taxonomy_name = self.data_files[idx]["taxonomy_name"]
        split_name = self.data_files[idx]["split_name"]
        image_paths = self.data_files[idx]["image_paths"]
        volume_path = self.data_files[idx]["volume_path"]

        # get the image renderings
        if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_paths = [
                image_paths[i]
                for i in random.sample(range(len(image_paths)), self.n_views_rendering)
            ]
        else:
            selected_rendering_paths = [
                image_paths[i] for i in range(self.n_views_rendering)]

        images = []
        for path in selected_rendering_paths:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(
                np.float32) / 255.

            if len(img.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                      (dt.now(), path))
                sys.exit(2)

            images.append(img)

        # get the volume rendering
        # ext = os.path.splitext(volume_path)[1]
        # if ext == ".binvox":
        with open(volume_path, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)

        return (taxonomy_name, split_name, np.asarray(images), volume)


class ShapeNetDataLoader:
    def __init__(self, args):
        self.dataset_taxonomy = None
        self.image_path = "data/ShapeNet/ShapeNetRendering"
        self.volume_path = "data/ShapeNet/ShapeNetVox32"

        with open("datasets/ShapeNet.json", encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def load_dataset(self, dataset_type, n_views_rendering):
        data_files = []

        tax_shortened = [self.dataset_taxonomy[i] for i in range(3)]

        # load the data for each taxonomy (aka each category)
        for taxonomy in tax_shortened:
            folder_name = taxonomy["taxonomy_id"]

            print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' %
                  (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))

            splits = []

            if dataset_type == DatasetType.TRAIN:
                splits = taxonomy["train"]
            elif dataset_type == DatasetType.VAL:
                splits = taxonomy["val"]
            elif dataset_type == DatasetType.TEST:
                splits = taxonomy["test"]

            # adds each of the taxonomy files to the list of data files
            taxonomy_files = self.get_files_for_taxonomy(
                folder_name, splits)
            data_files.extend(taxonomy_files)

            print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (
                dt.now(), len(data_files)))
        return ShapeNetDataset(dataset_type, data_files, n_views_rendering)

    def get_files_for_taxonomy(self, taxonomy_folder_name, splits):
        data_files = []

        for split in splits:
            vol_path = os.path.join(
                self.volume_path, taxonomy_folder_name, split, "model.binvox")
            if not os.path.exists(vol_path):
                print('[WARN] %s Ignore sample %s/%s since volume file does not exist.' %
                      (dt.now(), taxonomy_folder_name, split))
                continue

            # get file paths for images
            img_paths = []
            img_file_path = os.path.join(
                self.image_path, taxonomy_folder_name, split, "rendering")
            views = sorted(os.listdir(img_file_path))
            # the folder also contains rendering.txt and metadata (which we don't want)
            views = views[:len(views)-2]

            for view in views:
                curr_file_path = os.path.join(
                    self.image_path, taxonomy_folder_name, split, "rendering", view)
                if not os.path.exists(curr_file_path):
                    continue

                img_paths.append(curr_file_path)

            if len(img_paths) == 0:
                print('[WARN] %s Ignore sample %s/%s since image files do not exist.' %
                      (dt.now(), taxonomy_folder_name, split))
                continue

            # append the paths for the taxonomy to the output
            data_files.append({
                "taxonomy_name": taxonomy_folder_name,
                "split_name": split,
                "image_paths": img_paths,
                "volume_path": vol_path
            })

        return data_files


######################################################################################


DATASET_MAPPINGS = {
    "ShapeNet": ShapeNetDataLoader
}
