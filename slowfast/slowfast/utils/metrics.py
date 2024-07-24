#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""
import numpy as np
import torch
from sklearn.metrics import f1_score


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topks_correct_multi_label(preds, labels, ks):
    """
    Computes the number of top-k correct predictions for multi-label classification.

    Args:
        preds (tensor): Predictions with probabilities or scores, shape (N, ClassNum).
        labels (tensor): Ground truth labels, shape (N, ClassNum), where each element is binary.
        ks (list): List of top-k values to compute.

    Returns:
        topks_correct (list): List of numbers, where the `i`-th entry corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"

    # Find the top max_k predictions for each sample
    _, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)

    # For multi-label classification, we need to check whether all relevant labels are in the top-k predictions
    topks_correct = []
    for k in ks:
        top_k_preds = top_max_k_inds[:, :k]
        correct = torch.stack([torch.any(labels[i][top_k_preds[i]].bool()) for i in range(labels.size(0))])
        topks_correct.append(correct.sum().item())

    return topks_correct


def topk_errors_multi_label(preds, labels, ks):
    """
    Computes the top-k error for multi-label classification.

    Args:
        preds (tensor): Predictions with probabilities or scores, shape (N, ClassNum).
        labels (tensor): Ground truth labels, shape (N, ClassNum), where each element is binary.
        ks (list): List of ks to calculate the top accuracies.

    Returns:
        topk_errors (list): List of top-k errors for each k.
    """
    num_topks_correct = topks_correct_multi_label(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies_multi_label(preds, labels, ks):
    """
    Computes the top-k accuracy for multi-label classification.

    Args:
        preds (tensor): Predictions with probabilities or scores, shape (N, ClassNum).
        labels (tensor): Ground truth labels, shape (N, ClassNum), where each element is binary.
        ks (list): List of ks to calculate the top accuracies.

    Returns:
        topk_accuracies (list): List of top-k accuracies for each k.
    """
    num_topks_correct = topks_correct_multi_label(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def f1_scores_multi_label(preds, labels, average='macro', threshold=0.5):
    """
    Computes the F1 score for multi-label classification.

    Args:
        preds (tensor): Predictions with probabilities or scores, shape (N, ClassNum).
        labels (tensor): Ground truth labels, shape (N, ClassNum), where each element is binary.
        average (str): Averaging method for F1 score. Options are 'micro', 'macro', 'weighted', or 'samples'.

    Returns:
        f1_score (float): F1 score for multi-label classification.
    """
    # Convert tensors to numpy arrays
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    preds = np.array(preds)
    labels = np.array(labels)

    preds = (preds >= threshold).astype(int)

    zero_division = 0

    # Binarize predictions
    f1 = f1_score(labels, preds, average=average, zero_division=zero_division)

    return f1
