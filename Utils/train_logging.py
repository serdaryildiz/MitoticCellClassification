import argparse
import logging
import os
import yaml

from torch import nn
from torch.utils.tensorboard import SummaryWriter


def over_write_args(args: argparse.Namespace, yml: [dict, str]) -> None:
    """
        overwrites arguments for given dict or yaml file path.
    :param args: arguments
    :param yml: yaml file path or dict
    :return: None
    """
    if type(yml) == str:
        if yml == '':
            return
        with open(yml, 'r', encoding='utf-8') as f:
            dic = yaml.load(f.read(), Loader=yaml.Loader)
            for k in dic:
                setattr(args, k, dic[k])
    elif type(yaml) == dict:
        for k in yml:
            setattr(args, k, yml[k])
    else:
        raise Exception(f"Unknown parameter for overwrite : {yml}")


def count_parameters(model: nn.Module) -> int:
    """
        counts parameters for given pytorch module
    :param model: pytorch model
    :return: number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(name: str = None, save_path: str = None, level: str = 'INFO') -> logging.Logger:
    """
        create logger instance
    :param name: logger name
    :param save_path: log save path
    :param level: logging level
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # set time format
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    # if save file is given
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


class TBLog:
    """
    Tensorboard logger
    """
    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, update_dict: dict, it: int, suffix: str = None) -> None:
        """
            update tensorboard
        :param update_dict: key and values for logging
        :param it: iteration
        :param suffix: suffix for parsing keys
        :return: None
        """
        if suffix is None:
            suffix = ''
        for key, value in update_dict.items():
            self.writer.add_scalar(suffix + key, value, it)
