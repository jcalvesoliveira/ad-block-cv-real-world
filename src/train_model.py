import typer
from mrcnn.config import Config
from mrcnn.utils import compute_ap
from numpy import expand_dims, mean
from mrcnn.model import MaskRCNN, load_image_gt, mold_image

from dataset import AdvertisementDataset


class AdsConfig(Config):
    NAME = "ads_cfg"
    NUM_CLASSES = 4
    STEPS_PER_EPOCH = 131
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 400
    IMAGE_MAX_DIM = 512


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, cfg, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                 r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    mAP = mean(APs)
    return mAP


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
    # evaluate model on training dataset
    train_mAP = evaluate_model(train_set, model, config)
    print("Train mAP: %.3f" % train_mAP)
    # evaluate model on test dataset
    test_mAP = evaluate_model(test_set, model, config)
    print("Test mAP: %.3f" % test_mAP)

    # save model
    model.keras_model.save_weights('models/mask_rcnn_ads_cfg.h5')


if __name__ == '__main__':
    typer.run(main)