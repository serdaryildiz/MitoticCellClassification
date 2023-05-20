import os
import random
from typing import Tuple

import numpy
import torch
import torch.backends.cudnn as cudnn

import argparse

from Datasets.baseDataset import BaseDataset
from trainer import Trainer
from Utils import get_logger, over_write_args, getModel, getTransforms


def __getDatasets(dataset_arguments: dict, task) -> BaseDataset:
    """
        returns test dataset based on task
    :param dataset_arguments: dataset arguments
    :return: datasets
    """
    # get Transforms
    _, test_transforms = getTransforms(dataset_arguments["transforms"])
    dataset_name = dataset_arguments["name"].lower()
    if dataset_name == 'basedataset':
        if task == "train":
            test_dataset = BaseDataset(dataset_root=dataset_arguments["train_root"],
                                       transforms=test_transforms,
                                       class_names=dataset_arguments["class_names"])
        elif task == "val":
            test_dataset = BaseDataset(dataset_root=dataset_arguments["val_root"],
                                       transforms=test_transforms,
                                       class_names=dataset_arguments["class_names"])
        elif task == "test":
            test_dataset = BaseDataset(dataset_root=dataset_arguments["test_root"],
                                       transforms=test_transforms,
                                       class_names=dataset_arguments["class_names"])
        else:
            raise Exception("Unknown Test")
    else:
        raise Exception("Unknown Dataset name!")
    return test_dataset


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

    # init trainer
    trainer = Trainer(args=opt, tb_logger=None, logger=logger)

    # init dataset
    logger.info(f"Task : {opt.task}")
    test_dataset = __getDatasets(opt.dataset, opt.task)

    # get model
    model = getModel(opt.model)

    # set model
    trainer.setModel(model)
    trainer.load_model(opt.save_path, "model_best.pth")

    eval_dict = trainer.evaluate(evalDataset=test_dataset)
    logger.info(f"Test Results : {eval_dict}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Base Classification Project!')
    parser.add_argument('--config', type=str, default='./configs/base.yaml')
    parser.add_argument('--task', type=str, default='test')
    args = parser.parse_args()
    over_write_args(args, args.config)
    main_worker(args)
