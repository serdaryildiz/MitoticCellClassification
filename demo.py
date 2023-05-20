import glob
import os
import random
from typing import Tuple

import cv2
import numpy
import torch
import torch.backends.cudnn as cudnn

import argparse

from PIL import Image

from Datasets.baseDataset import BaseDataset
from trainer import Trainer
from Utils import get_logger, over_write_args, getModel, getTransforms


def __load_model(model, load_dir, load_name):
    """
        loads model
    """
    load_path = os.path.join(load_dir, load_name)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])


@torch.no_grad()
def main_worker(opt):
    # random seed
    assert opt.seed is not None
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    numpy.random.seed(opt.seed)
    cudnn.deterministic = True

    # set experiment save path
    save_path = os.path.join(opt.save_dir, opt.save_name)
    opt.save_path = save_path
    if not os.path.exists(save_path):
        raise Exception('experiment is not exists: {}'.format(save_path))

    # set logger
    logger_level = "INFO"
    logger = get_logger(opt.save_name, None, logger_level)
    logger.warning(f"USE GPU: {opt.gpu} for training")
    logger.info(opt)
    device = torch.device(opt.gpu)

    # get model
    model = getModel(opt.model)
    _, test_transforms = getTransforms(opt.dataset["transforms"])

    # set model
    __load_model(model, opt.save_path, "model_best.pth")
    model.eval()
    model.to(device)

    # get images
    image_paths = glob.glob(os.path.join(opt.images_root, "*"))

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        batch = test_transforms(image)[None, ...].to(device)
        output = model(batch)
        pred = torch.max(output, dim=1)[1][0].cpu().item()

        logger.info(f"image : {image_path.split('/')[-1]} -> {pred}")
        cv2.imshow("image", numpy.array(image)[..., ::-1])
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Base Classification Project!')
    parser.add_argument('--config', type=str, default='./configs/base.yaml')
    parser.add_argument('--images-root', type=str, default='tests/TestDataset/a-class')
    args = parser.parse_args()
    over_write_args(args, args.config)
    main_worker(args)
