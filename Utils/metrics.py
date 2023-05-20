from typing import Tuple

import numpy
import torch

from sklearn.metrics import precision_recall_fscore_support


def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-6) -> Tuple[float, float, float]:
    """
        calculate dice coefficient for binary maps
    :param y_true: labels (0, 1)
    :param y_pred: predictions (0, 1)
    :param epsilon: epsilon for zero division error
    :return: dice coefficient, intersection, union
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    dice = (2. * intersection) / (union + epsilon)
    return dice.detach().cpu().item(), intersection.cpu().item(), union.cpu().item()


def dice_coef_multilabel(y_true: torch.Tensor, y_pred: torch.Tensor, numLabels: int) -> dict:
    """
        calculate dice coefficient for multi label
    :param y_true: labels
    :param y_pred: predictions
    :param numLabels: number of labels
    :return: dictionary of dice coefficient
    """
    dice = []
    # for all classes
    for index in range(numLabels):
        y_true_maks = torch.where(y_true == index, 1, 0)
        y_pred_maks = torch.where(y_pred == index, 1, 0)
        coef, intersection, union = dice_coef(y_true_maks, y_pred_maks, epsilon=0)
        dice.append(coef)
    return_dict = {"Dice/Acc": numpy.nanmean(dice)}
    for i, d in enumerate(dice):
        return_dict.update({f"Dice/Acc {i}": d})
    return return_dict


def intersection_and_union(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float]:
    """
        calculates intersection and union values for binary maps
    :param y_true: labels
    :param y_pred: predictions
    :return: intersection, union
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(torch.where((y_true_f + y_pred_f) > 0, 1, 0))
    return intersection.detach().cpu().item(), union.detach().cpu().item()


def iou_multilabel(y_true: torch.Tensor, y_pred: torch.Tensor, numLabels: int) -> dict:
    """
        calculates intersection over union for multi label
    :param y_true: labels
    :param y_pred: predictions
    :param numLabels: number of labels
    :return: dictionary of iou metric
    """
    iou = []
    # for all classes
    for index in range(numLabels):
        y_true_maks = torch.where(y_true == index, 1, 0)
        y_pred_maks = torch.where(y_pred == index, 1, 0)
        intersection, union = intersection_and_union(y_true_maks, y_pred_maks)
        if union == 0:
            iou.append(numpy.nan)
        else:
            iou.append(intersection / union)
    return_dict = {"IoU/Total": numpy.nanmean(iou)}
    for i, d in enumerate(iou):
        return_dict.update({f"IoU {i}": d})
    return return_dict


def getAccRecallPrecision(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float, float]:
    """
        calculates recall and precision scores for binary map
    :param y_true: labels
    :param y_pred: predictions
    :return: accuracy, precision, recall
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    acc = torch.sum(y_true_f * y_pred_f) / y_pred_f.size(0)

    y_true_f = y_true_f.tolist()
    y_pred_f = y_pred_f.tolist()

    precision, recall, _, __ = precision_recall_fscore_support(y_true_f, y_pred_f, labels=[1])
    # weight = numpy.sum(y_true_f) / len(y_true_f)
    return acc.detach().cpu().item(), precision[0], recall[0]


def pixelBasedMetrics_multilabel(y_true: torch.Tensor, y_pred: torch.Tensor, numLabels: int) -> dict:
    """
        calculates pixel based metrics which are accuracy precision and recall for multi label
    :param y_true: labels
    :param y_pred: prediction
    :param numLabels: number of labels
    :return: dictionary of precision recall and accuracy
    """
    acc = []
    recall = []
    precision = []
    return_dict = {}
    for index in range(numLabels):
        y_true_maks = torch.where(y_true == index, 1, 0)
        y_pred_maks = torch.where(y_pred == index, 1, 0)
        acc_, precision_, recall_ = getAccRecallPrecision(y_true_maks, y_pred_maks)
        return_dict[f"Acc/{index}"] = acc_
        return_dict[f"precision/{index}"] = precision_
        return_dict[f"recall/{index}"] = recall_
        acc.append(acc_)
        precision.append(precision_)
        recall.append(recall_)
    return return_dict


def classBasedMetrics_PrecisionRecall(y_true: torch.Tensor, y_pred: torch.Tensor, numLabels: int) -> dict:
    """
        calculates precision and recall for multi label
    :param y_true: labels
    :param y_pred: prediction
    :param numLabels: number of labels
    :return: dictionary of precision recall and accuracy
    """
    recall = []
    precision = []
    return_dict = {}
    for index in range(numLabels):
        y_true_maks = torch.where(y_true == index, 1, 0)
        y_pred_maks = torch.where(y_pred == index, 1, 0)
        _, precision_, recall_ = getAccRecallPrecision(y_true_maks, y_pred_maks)
        return_dict[f"precision/{index}"] = precision_
        return_dict[f"recall/{index}"] = recall_
        precision.append(precision_)
        recall.append(recall_)
    return return_dict
