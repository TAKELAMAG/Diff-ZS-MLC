import os
from .coco_detection_synthetic_image import CocoDetection
from .pascal_voc import voc2007

MODEL_TABLE = {
    'coco': CocoDetection,
    #'voc2007': voc2007,
}


def build_dataset_synthetic_image(cfg, data_split, annFile=""):
    print(' -------------------- Building Dataset ----------------------')
    print('DATASET.ROOT = %s' % cfg.DATASET.ROOT)
    print('data_split = %s' % data_split)
    print('PARTIAL_PORTION= %f' % cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)
    if annFile != "":
        annFile = os.path.join(cfg.DATASET.ROOT, 'syn_annotations', annFile)
    try:
        if 'train' in data_split or 'Train' in data_split:
            img_size = cfg.INPUT.TRAIN.SIZE[0]
        else:
            img_size = cfg.INPUT.TEST.SIZE[0]
    except:
        img_size = cfg.INPUT.SIZE[0]
    print('INPUT.SIZE = %d' % img_size)

    return MODEL_TABLE[cfg.DATASET.NAME](cfg.DATASET.ROOT, data_split, img_size,
                                         p=cfg.DATALOADER.TRAIN_X.PORTION, annFile=annFile,
                                         label_mask=cfg.DATASET.MASK_FILE,
                                         partial=cfg.DATALOADER.TRAIN_X.PARTIAL_PORTION)
