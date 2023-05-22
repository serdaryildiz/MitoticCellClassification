from typing import Tuple, Optional
import os

import math
import numpy
# import albumentations
# from albumentations.pytorch import ToTensorV2
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

import Datasets
import Models
import LossFunctions
from LossFunctions import BaseLoss
from Datasets.baseDataset import BaseDataset


def getModel(model_arguments: dict) -> nn.Module:
    """
        gets model name and return model instance
    :param model_arguments: model arguments
    :return: torch model
    """
    model_name = model_arguments["name"]
    model_class = getattr(Models, model_name)
    model_arguments.pop("name")

    w = model_arguments.pop("weights", False)
    if w:
        base_model: nn.Module = model_class(**{"weights": w})
        model_arguments["weights"] = None
        model: nn.Module = model_class(**model_arguments)

        pretrain_dict = base_model.state_dict()
        model_dict = model.state_dict()
        new_state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in zip(model_dict.keys(), pretrain_dict.values())}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model = model_class(**model_arguments)
    return model


def getOptimizer(optimizer_arguments: dict, model: nn.Module) -> torch.optim.Optimizer:
    """
        gets optimizer
    :param optimizer_arguments:
    :param model: pytorch model
    :return: optimizer
    """

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if 'bn' in name or 'bias' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    optimizer_name = optimizer_arguments["name"].lower()
    lr = float(optimizer_arguments["lr"])
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=float(optimizer_arguments["momentum"]),
                                    weight_decay=float(optimizer_arguments["weight_decay"]), nesterov=True)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=float(optimizer_arguments["weight_decay"]))
    else:
        raise Exception(f"Unknown optimizer name! : {optimizer_name}")
    return optimizer


def getScheduler(scheduler_arguments: dict, optimizer: torch.optim.Optimizer):
    """
        gets scheduler
    :param scheduler_arguments: arguments
    :param optimizer: optimizer
    :return: scheduler or None
    """
    if scheduler_arguments is None:
        return None
    scheduler_name = scheduler_arguments["name"].lower()

    if scheduler_name is None:
        return None
    else:
        max_iter = scheduler_arguments["max-iter"]
        if scheduler_name == "cosine":
            return get_cosine_schedule_with_warmup(optimizer,
                                                   # args.iter * args.epoch,
                                                   max_iter,
                                                   num_warmup_steps=scheduler_arguments["num_warmup_steps"])
        elif scheduler_name == "onecyclelr":
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, scheduler_arguments["lr"], scheduler_arguments["max-iter"],
                                                       pct_start=0.0075, cycle_momentum=False)
        else:
            raise Exception(f"Unknown scheduler name : {scheduler_name}")


def getDatasets(dataset_arguments: dict) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
    """
        returns train, validation and test datasets
    :param dataset_arguments: dataset arguments
    :return: datasets
    """
    # get Transforms
    train_transforms, test_transforms = getTransforms(dataset_arguments["transforms"])
    dataset_name = dataset_arguments["name"].lower()
    if dataset_name == 'basedataset':
        trainDataset = BaseDataset(dataset_root=dataset_arguments["train_root"],
                                   transforms=train_transforms,
                                   class_names=dataset_arguments["class_names"])
        valDataset = BaseDataset(dataset_root=dataset_arguments["val_root"],
                                 transforms=test_transforms,
                                 class_names=dataset_arguments["class_names"])
        testDataset = BaseDataset(dataset_root=dataset_arguments["test_root"],
                                  transforms=test_transforms,
                                  class_names=dataset_arguments["class_names"])
    else:
        raise Exception("Unknown Dataset name!")

    return trainDataset, valDataset, testDataset


def getLossFunction(loss_arguments: dict) -> BaseLoss:
    """
        get loss function
    :param loss_arguments: loss function arguments
    :return: function
    """
    loss_name = loss_arguments["name"]
    loss_class = getattr(LossFunctions, loss_name)
    loss_arguments.pop("name")
    return loss_class(**loss_arguments)


def getTransforms(transforms_arguments):
    """
        gets train and test transforms
    :param transforms_arguments: transform arguments
    :return: transforms
    """
    # train_transform = albumentations.Compose(
    #     [
    #         albumentations.Resize(height=transforms_arguments["height"], width=transforms_arguments["width"]),
    #         albumentations.Rotate(limit=35, p=1.0),
    #         albumentations.HorizontalFlip(p=0.5),
    #         albumentations.VerticalFlip(p=0.1),
    #         albumentations.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )
    #
    # val_transforms = albumentations.Compose(
    #     [
    #         albumentations.Resize(height=transforms_arguments["height"], width=transforms_arguments["width"]),
    #         albumentations.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    train_transform = transforms.Compose([
        transforms.Resize((transforms_arguments["height"],
                           transforms_arguments["width"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((transforms_arguments["height"],
                           transforms_arguments["width"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transforms


def saveIdxs(train_idxs: numpy.ndarray, val_idxs: numpy.ndarray, test_idxs: numpy.ndarray, root: str) -> None:
    """
        save indexes
    :param train_idxs: train indexes
    :param val_idxs: validation indexes
    :param test_idxs: test indexes
    :param root: save root dir
    :return: None
    """
    train_idxs_path = os.path.join(root, "train_idxs.npy")
    test_idxs_path = os.path.join(root, "test_idxs.npy")
    val_idxs_path = os.path.join(root, "val_idxs.npy")
    # assert os.path.exists(train_idxs_path) is False
    # assert os.path.exists(test_idxs_path) is False
    numpy.save(train_idxs_path, train_idxs)
    numpy.save(val_idxs_path, val_idxs)
    numpy.save(test_idxs_path, test_idxs)
    return


def loadIdxs(load_path: str) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
        loads train, validation and test indexes
    :param load_path: load dir root
    :return: indexes
    """
    train_idxs_path = os.path.join(load_path, "train_idxs.npy")
    val_idxs_path = os.path.join(load_path, "val_idxs.npy")
    test_idxs_path = os.path.join(load_path, "test_idxs.npy")
    assert os.path.exists(train_idxs_path) is True
    assert os.path.exists(val_idxs_path) is True
    assert os.path.exists(test_idxs_path) is True
    train_idxs = numpy.load(train_idxs_path)
    val_idxs = numpy.load(val_idxs_path)
    test_idxs = numpy.load(test_idxs_path)
    return train_idxs, val_idxs, test_idxs


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    """
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    """

    def _lr_lambda(current_step):
        """
        _lr_lambda returns a multiplicative factor given an integer parameter epochs.
        Decaying criteria: last_epoch
        """
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
