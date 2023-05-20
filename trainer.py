import logging
import os
from argparse import Namespace

import numpy
import torch
import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, RandomSampler, Dataset

from LossFunctions import BaseLoss
from Utils import TBLog, classBasedMetrics_PrecisionRecall


class Trainer:
    def __init__(self, args: Namespace, tb_logger: TBLog = None, logger: logging.Logger = None):
        self.args = args

        self.gpu = torch.device(args.gpu)
        self.model = None
        self.it = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.num_classes = self.args.model["num_classes"]

        # init dataset
        self.trainDataset = None
        self.trainDataloader = None
        self.evalDataset = None
        self.evalDataloader = None

        # optimizer and scheduler
        self.scheduler = None
        self.optimizer = None

        # loss
        self.loss_fn = None
        self.weight = None

        # logging
        self.tb_log = tb_logger
        self.print_fn = print if logger is None else logger.info
        return

    def train(self):
        """
            main train loop
        """
        self.model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        if self.args.resume:
            eval_dict = self.evaluate()
            print(eval_dict)

        start_batch.record()
        tbar = tqdm.tqdm(total=len(self.trainDataloader), colour='BLUE')
        for _, samples, targets in self.trainDataloader:
            tbar.update(1)
            self.it += 1

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            samples, targets = samples.to(self.gpu), targets.to(self.gpu).long()

            outputs = self.model(samples)

            loss = self.loss_fn(outputs, targets)

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {'train/loss': loss.detach().cpu().item(), 'lr': self.optimizer.param_groups[0]['lr'],
                       'train/prefetch_time': start_batch.elapsed_time(end_batch) / 1000.,
                       'train/run_time': start_run.elapsed_time(end_run) / 1000.}

            if self.it % self.args.num_eval_iter == 0:
                eval_dict = self.evaluate()
                tb_dict.update(eval_dict)
                save_path = self.args.save_path
                if tb_dict['eval/Acc'] > self.best_eval_acc:
                    self.best_eval_acc = tb_dict['eval/Acc']
                    self.best_it = self.it

                self.print_fn(
                    f"\n {self.it} iteration, {tb_dict}, \n BEST_EVAL_ACC: {self.best_eval_acc}, at {self.best_it} iters")
                self.print_fn(
                    f" {self.it} iteration, ACC: {tb_dict['eval/Acc']}\n")
                if self.it == self.best_it:
                    self.save_model('model_best.pth', save_path)

            if self.tb_log is not None:
                self.tb_log.update(tb_dict, self.it)
            del tb_dict
            start_batch.record()

        eval_dict = self.evaluate()
        eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it})
        return eval_dict

    def save_model(self, save_name, save_path):
        """
            saves model
        """
        save_filename = os.path.join(save_path, save_name)
        self.model.eval()
        save_dict = {"model": self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                     'it': self.it}
        torch.save(save_dict, save_filename)
        self.model.train()
        self.print_fn(f"model saved: {save_filename}\n")

    def load_model(self, load_dir, load_name):
        """
            loads model
        """
        load_path = os.path.join(load_dir, load_name)
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model'])
        if checkpoint['optimizer'] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn(f'model loaded from {load_path}')

    @torch.no_grad()
    def evaluate(self, model=None, evalDataset: Dataset = None):
        """
            evaluate model for given dataset
        """
        self.print_fn("\n Evaluation!!!")

        if model is None:
            model = self.model
        if evalDataset is not None:
            evalDataloader = DataLoader(evalDataset, self.args.eval_batch_size, shuffle=False, num_workers=0)
        else:
            evalDataloader = self.evalDataloader

        num_correct = 0
        num_sample = 0
        model.eval()
        preds_arr = None
        targets_arr = None
        for _, samples, targets in evalDataloader:
            samples, targets = samples.to(self.gpu), targets.to(self.gpu)
            outputs = model(samples)
            preds = torch.max(outputs, dim=1)[1]

            if preds_arr is None:
                preds_arr = preds.detach()
                targets_arr = targets.detach()
            else:
                preds_arr = torch.concat((preds_arr, preds.detach()))
                targets_arr = torch.concat((targets_arr, targets.detach()))

            num_correct += (preds == targets).sum()
            num_sample += torch.numel(preds)

        eval_dict = classBasedMetrics_PrecisionRecall(targets_arr, preds_arr, self.num_classes)
        self.print_fn("Confusion Matrix :\n"+numpy.array2string(confusion_matrix(targets_arr.cpu().numpy(), preds_arr.cpu().numpy()))+"\n")
        eval_dict["eval/Acc"] = (num_correct / num_sample).cpu().item()
        model.train()
        return eval_dict

    def set_optimizer(self, optimizer, scheduler=None):
        """
            sets optimizer and scheduler
        """
        self.optimizer = optimizer
        self.scheduler = scheduler

    def setModel(self, model):
        """
            sets model class
        """
        self.model = model.cuda(self.gpu)

    def setDatasets(self, trainDataset, evalDataset):
        """
            set train and evaluation dataset and also initiate dataset loaders
        :param trainDataset: Train dataset class
        :param evalDataset: evaluation dataset
        """
        self.trainDataset = trainDataset
        self.evalDataset = evalDataset

        self.trainDataloader = DataLoader(trainDataset, batch_size=self.args.batch_size,
                                          sampler=RandomSampler(data_source=trainDataset,
                                                                replacement=True,
                                                                num_samples=self.args.iter * self.args.batch_size),
                                          num_workers=self.args.num_workers, drop_last=True)
        self.evalDataloader = DataLoader(evalDataset, self.args.eval_batch_size, shuffle=False, num_workers=self.args.num_workers)

    def setLoss(self, loss_fn: BaseLoss):
        """
            sets loss function
        """
        self.loss_fn = loss_fn
