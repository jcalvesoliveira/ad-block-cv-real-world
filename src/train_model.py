import typer
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

from .dataset import AdvertisementDataset


class AdsConfig(Config):
    NAME = "ads_cfg"
    NUM_CLASSES = 3
    STEPS_PER_EPOCH = 131

def main():
    # train set
    train_set = AdvertisementDataset()
    train_set.load_dataset('data/raw', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # test/val set
    test_set = AdvertisementDataset()
    test_set.load_dataset('data/raw', is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    # prepare config
    config = AdsConfig()
    config.display()
    # define the model
    model = MaskRCNN(mode='training', model_dir='models/', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('models/mask_rcnn_coco.h5',
                       by_name=True,
                       exclude=[
                           "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
                           "mrcnn_mask"
                       ])
    # train weights (output layers or 'heads')
    model.train(train_set,
                test_set,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')



if __name__ == '__main__':
    typer.run(main)