import os
import random

import numpy
import torch
import torch.backends.cudnn as cudnn

import argparse

from trainer import Trainer
from Utils import TBLog, get_logger, over_write_args, getDatasets, getOptimizer, getScheduler, getModel, getLossFunction


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
    if os.path.exists(save_path) and not opt.overwrite:
        raise Exception('already existing model: {}'.format(save_path))

    # set logger
    tb_logger = TBLog(save_path, 'tensorboard')
    logger_level = "INFO"
    logger = get_logger(opt.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {opt.gpu} for training")
    logger.info(opt)

    # init trainer
    trainer = Trainer(args=opt, tb_logger=tb_logger, logger=logger)

    # init dataset
    trainDataset, valDataset, testDataset = getDatasets(opt.dataset)
    logger.info("Train Class Names : ", list(trainDataset.get_name2id_map().keys()), "Sample Size : ", len(trainDataset))
    logger.info("Val Class Names : ", list(valDataset.get_name2id_map().keys()), "Sample Size : ", len(valDataset))
    logger.info("Test Class Names : ", list(testDataset.get_name2id_map().keys()), "Sample Size : ", len(testDataset))
    trainer.setDatasets(trainDataset=trainDataset, evalDataset=valDataset)

    # get model
    model = getModel(opt.model)

    # set optimizer and scheduler
    optimizer = getOptimizer(opt.optimizer, model)
    opt.scheduler["max-iter"] = opt.iter * opt.epoch
    opt.scheduler["lr"] = opt.optimizer["lr"]
    scheduler = getScheduler(opt.scheduler, optimizer=optimizer)

    trainer.set_optimizer(optimizer=optimizer, scheduler=scheduler)

    # set loss function
    loss_fn = getLossFunction(opt.loss)
    trainer.setLoss(loss_fn=loss_fn)

    # set model
    trainer.setModel(model)

    # If resume training
    if opt.resume:
        model.load_model(opt.load_path)

    # start training
    for e in range(opt.epoch):
        trainer.train()

    # test with best model
    if testDataset is not None:
        trainer.load_model(opt.save_path, "model_best.pth")
        eval_dict = trainer.evaluate(evalDataset=testDataset)
        logger.info(f"Test Results : {eval_dict}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Base Classification Project!')
    parser.add_argument('--config', type=str, default='./configs/base.yaml')
    args = parser.parse_args()
    over_write_args(args, args.config)
    main_worker(args)
