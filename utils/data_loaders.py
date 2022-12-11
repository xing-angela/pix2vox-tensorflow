from PIL import Image
import cv2
import random
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils.binvox_rw
from datetime import datetime as dt


###################################### SHAPENET ######################################


class ShapeNetDataset:
    def __init__(self, dataset_type, files, n_views_rendering):
        self.dataset_type = dataset_type
        self.file_list = files
        self.n_views_rendering = n_views_rendering

        self.taxonomy_names = []
        self.sample_names = []
        self.images = []
        self.vols = []

        # loads the data on initialization
        self.load_data()

    def load_data(self):
        for file in self.file_list:
            # Saves the taxonomy names and sample names
            taxonomy_name = file['taxonomy_name']
            sample_name = file['sample_name']
            self.taxonomy_names.append(taxonomy_name)
            self.sample_names.append(sample_name)

            image_paths = file['image_paths']
            volume_path = file['volume_path']

            # Get the image renderings
            if self.dataset_type == 'train':
                selected_rendering_paths = [
                    image_paths[i]
                    for i in random.sample(range(len(image_paths)), self.n_views_rendering)
                ]
            else:
                selected_rendering_paths = [
                    image_paths[i] for i in range(self.n_views_rendering)
                ]

            images = []
            for path in selected_rendering_paths:
                # Handles the image loading
                with Image.open(path) as img:
                    img = np.array(img.resize((224, 224)))[:, :, :3]

                if len(img.shape) < 3:
                    print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                          (dt.now(), path))
                    sys.exit(2)

                images.append(img)
            self.images.append(images)

            # Handles the volume loading
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

            self.vols.append(volume)

        self.images = np.asarray(self.images)
        self.vols = np.asarray(self.vols)

        print('[INFO] %s Complete loading files of the dataset.' % (dt.now()))


class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.image_path = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.volume_path = cfg.DATASETS.SHAPENET.VOXEL_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def load_dataset_files(self, dataset_type, n_views_rendering):
        files = []

        # Load data for each category
        for taxonomy in [self.dataset_taxonomy[0]]:  # 4 for chair
            taxonomy_folder_name = taxonomy['taxonomy_id']
            print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' %
                  (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))

            samples = []
            if dataset_type == 'train':
                samples = taxonomy['train']
            elif dataset_type == 'test':
                samples = taxonomy['train']
            elif dataset_type == 'val':
                samples = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(
                taxonomy_folder_name, samples))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (
            dt.now(), len(files)))

        return ShapeNetDataset(dataset_type, files, n_views_rendering)

    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []

        for sample_name in samples:
            # Get file path of volumes
            vol_path = os.path.join(
                self.volume_path, taxonomy_folder_name, sample_name, 'model.binvox')
            if not os.path.exists(vol_path):
                print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file path for images
            img_paths = []
            img_file_path = os.path.join(
                self.image_path, taxonomy_folder_name, sample_name, 'rendering')
            views = sorted(os.listdir(img_file_path))

            for view in views:
                try:
                    int(view[0])
                    curr_file_path = os.path.join(
                        self.image_path, taxonomy_folder_name, sample_name, 'rendering', view)
                    if not os.path.exists(curr_file_path):
                        continue
                except ValueError:
                    continue

                img_paths.append(curr_file_path)

            if len(img_paths) == 0:
                print('[WARN] %s Ignore sample %s/%s since image files do not exist.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # append the paths for the taxonomy to the output
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'image_paths': img_paths,
                'volume_path': vol_path
            })

        return files_of_taxonomy

#################################### END SHAPENET ####################################

#################################### BEGIN PIX3D #####################################


class Pix3dDataset:
    def __init__(self, files):
        self.file_list = files

        self.taxonomy_names = []
        self.sample_names = []
        self.images = []
        self.vols = []
        self.bounding_boxes = []

        # load the data
        self.load_data()

    def load_data(self):
        for file in self.file_list:
            # Saves the taxonomy names and sample names
            taxonomy_name = file['taxonomy_name']
            sample_name = file['sample_name']
            self.taxonomy_names.append(taxonomy_name)
            self.sample_names.append(sample_name)

            image_paths = file['image_paths']
            volume_path = file['volume_path']

            bounding_box = file['bounding_box']
            self.bounding_boxes.append(bounding_box)

            # Get image renderings
            rendering_images = []
            for path in image_paths:
                # Handles the image loading
                with Image.open(path) as img:
                    img = np.array(img.resize((224, 224)))[:, :, :3]

                if len(img.shape) < 3:
                    print('[WARN] %s It seems the image file %s is grayscale.' % (
                        dt.now(), path))
                    img = np.stack((img, )*3, -1)

                rendering_images.append(img)
            self.images.append(rendering_images)

            # Get data of volume
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

            self.vols.append(volume)

        self.images = np.asarray(self.images)
        self.vols = np.asarray(self.vols)

        print('[INFO] %s Complete loading files of the dataset.' % (dt.now()))


class Pix3dDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.annotations = dict()
        self.volume_path_template = cfg.DATASETS.PIX3D.VOXEL_PATH
        self.rendering_image_path_template = cfg.DATASETS.PIX3D.RENDERING_PATH

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.PIX3D.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

        # Load all annotations of the dataset
        _annotations = None
        with open(cfg.DATASETS.PIX3D.ANNOTATION_PATH, encoding='utf-8') as file:
            _annotations = json.loads(file.read())

        for anno in _annotations:
            filename, _ = os.path.splitext(anno['img'])
            anno_key = filename[4:]
            self.annotations[anno_key] = anno

    def load_dataset_files(self, dataset_type, n_views_rendering):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_name = taxonomy['taxonomy_name']
            print('[INFO] %s Collecting files of Taxonomy[Name=%s]' %
                  (dt.now(), taxonomy_name))

            samples = []
            if dataset_type == 'train':
                samples = taxonomy['train']
            elif dataset_type == 'test':
                samples = taxonomy['test']
            elif dataset_type == 'val':
                samples = taxonomy['test']

            files.extend(self.get_files_of_taxonomy(taxonomy_name, samples))

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (
            dt.now(), len(files)))

        return Pix3dDataset(files)

    def get_files_of_taxonomy(self, taxonomy_name, samples):
        files_of_taxonomy = []

        for sample_name in samples:
            # Get image annotations
            anno_key = '%s/%s' % (taxonomy_name, sample_name)
            annotations = self.annotations[anno_key]

            # Get file list of rendering images
            _, img_file_suffix = os.path.splitext(annotations['img'])
            rendering_image_file_path = self.rendering_image_path_template % (taxonomy_name, sample_name,
                                                                              img_file_suffix[1:])

            # Get the bounding box of the image
            img_width, img_height = annotations['img_size']
            bbox = [
                annotations['bbox'][0] / img_width,
                annotations['bbox'][1] / img_height,
                annotations['bbox'][2] / img_width,
                annotations['bbox'][3] / img_height
            ]  # yapf: disable
            model_name_parts = annotations['voxel'].split('/')
            model_name = model_name_parts[2]
            volume_file_name = model_name_parts[3][:-
                                                   4].replace('voxel', 'model')

            # Get file path of volumes
            volume_file_path = self.volume_path_template % (
                taxonomy_name, model_name, volume_file_name)
            if not os.path.exists(volume_file_path):
                print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %
                      (dt.now(), taxonomy_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'bounding_box': bbox,
                'volume': volume_file_path,
            })

        return files_of_taxonomy

#################################### END PIX3D ####################################


DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    # 'Pascal3D': Pascal3dDataLoader,
    'Pix3D': Pix3dDataLoader
}
