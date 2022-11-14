import os
import random
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from mrcnn.utils import Dataset
from mrcnn.config import Config


class AdsConfig(Config):
    NAME = "ads_cfg"
    NUM_CLASSES = 4
    STEPS_PER_EPOCH = 131
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 400
    IMAGE_MAX_DIM = 512


class SignageConfig(AdsConfig):
    NAME = "sgn_cfg"
    NUM_CLASSES = 2


class AdvertisementDataset(Dataset):
    LABES_MAP = {'billboard': 1, 'signage': 2, 'branding': 3}

    def convert_annotations(self, annotations: pd.DataFrame) -> pd.DataFrame:
        data = []
        for index, row in annotations.iterrows():
            row_data = {}
            row_data['x1'] = row['coordinates'][0]['x']
            row_data['y1'] = row['coordinates'][0]['y']
            row_data['x2'] = row['coordinates'][1]['x']
            row_data['y2'] = row['coordinates'][1]['y']
            row_data['label'] = row['labels'][0]
            data.append(row_data)
        return pd.DataFrame(data)

    def has_annotation(self, annotation_path: str):
        boxes = pd.read_json(annotation_path, lines=True)
        boxes = self.convert_annotations(boxes)
        return boxes.empty is False

    @staticmethod
    def convert_bound_boxes_from_perc_to_loc(annotations: np.ndarray, w: int,
                                             h: int) -> np.ndarray:
        annotations[:, 0] = annotations[:, 0] * w
        annotations[:, 1] = annotations[:, 1] * h
        annotations[:, 2] = annotations[:, 2] * w
        annotations[:, 3] = annotations[:, 3] * h
        annotations[:, 0:4] = annotations[:, 0:4].astype(int)
        return annotations

    def image_path(self, dataset_dir: str, image_id: str) -> str:
        return dataset_dir + '/images/' + str(image_id) + '.jpg'

    def annotation_path(self, dataset_dir: str, image_id: str) -> str:
        return dataset_dir + '/annotations/' + str(image_id) + '.json'

    def load_dataset(self, dataset_dir, is_train=True, train_size=0.80):
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations/'
        image_files = os.listdir(images_dir)
        images_count = len(image_files)
        train_threshold = int(images_count * train_size)
        logging.info(f"Total images: {images_count}")
        for label in self.LABES_MAP:
            self.add_class("dataset", self.LABES_MAP[label], label)
        for i, filename in enumerate(os.listdir(images_dir)):
            if is_train and i >= train_threshold:
                continue
            if not is_train and i < train_threshold:
                continue
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = self.annotation_path(dataset_dir, image_id)
            if not self.has_annotation(ann_path):
                continue
            self.add_image('dataset',
                           image_id=image_id,
                           path=img_path,
                           annotation=ann_path)

    def extract_boxes(self, annotation_path: str,
                      image_path: str) -> Tuple[pd.DataFrame, int, int]:
        boxes = pd.read_json(annotation_path, lines=True)
        boxes = self.convert_annotations(boxes).values
        img = Image.open(image_path)
        width, height = img.size
        boxes = self.convert_bound_boxes_from_perc_to_loc(boxes, width, height)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotation_path = info['annotation']
        image_path = info['path']
        boxes, w, h = self.extract_boxes(annotation_path, image_path)
        annotations_count = boxes.shape[0]
        masks = np.zeros([h, w, annotations_count], dtype='uint8')
        class_ids = list()
        for i, box in enumerate(boxes):
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = self.LABES_MAP[box[4]]
            class_ids.append(self.class_names.index(box[4]))
        return masks, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class SignageDataset(AdvertisementDataset):
    LABES_MAP = {'ad': 1}

    def convert_annotations(self, annotations: pd.DataFrame) -> pd.DataFrame:
        data = []
        for index, row in annotations.iterrows():
            row_data = {}
            row_data['x1'] = row['coordinates'][0]['x']
            row_data['y1'] = row['coordinates'][0]['y']
            row_data['x2'] = row['coordinates'][1]['x']
            row_data['y2'] = row['coordinates'][1]['y']
            row_data['label'] = 'ad'
            data.append(row_data)
        return pd.DataFrame(data)