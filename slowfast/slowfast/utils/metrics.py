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
        preds (tensor): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (tensor): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    # Ensure all tensors are on the same device
    device = preds.device
    labels = labels.to(device)

    assert preds.size(0) == labels.size(0), "Batch dimension of predictions and labels must match"

    # Get the top max_k predictions for each sample
    max_k = max(ks)
    _, top_max_k_inds = torch.topk(preds, max_k, dim=1, largest=True, sorted=True)

    # Convert labels to one-hot encoding
    batch_size, num_classes = preds.size()
    labels_one_hot = torch.zeros(batch_size, num_classes, dtype=torch.bool, device=device)
    labels_one_hot[torch.arange(batch_size, device=device), labels] = True

    topks_correct = []
    for k in ks:
        # Create a mask for top-k predictions
        top_k_mask = torch.zeros_like(preds, dtype=torch.bool, device=device)
        for i in range(batch_size):
            top_k_mask[i, top_max_k_inds[i, :k]] = True

        # Compute the correct predictions
        correct_predictions = top_k_mask & labels_one_hot
        topk_correct_count = correct_predictions.any(dim=1).sum().item()
        topks_correct.append(topk_correct_count)

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


def topks_correct_binary(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value in binary classification.

    Args:
        preds (tensor): Array of predictions. Dimension is batch size N.
            For binary classification, this is a tensor of size N where each value
            is the probability or logit of the positive class.
        labels (tensor): Array of labels. Dimension is batch size N.
            Labels should be 0 or 1 for binary classification.
        ks (list): List of top-k values. For binary classification, this is typically [1].

    Returns:
        topks_correct (list): List of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    # Ensure all tensors are on the same device
    device = preds.device
    labels = labels.to(device).long()  # Ensure labels are long

    assert preds.size(0) == labels.size(0), "Batch dimension of predictions and labels must match"

    assert preds.size(0) == labels.size(0), "Batch dimension of predictions and labels must match"

    # Convert logits to probabilities if needed
    if preds.dim() == 1:
        preds = torch.sigmoid(preds)

    # Binarize predictions using a threshold of 0.5
    pred_labels = (preds > 0.5).long()

    correct_predictions = (pred_labels == labels).sum().item()

    return [correct_predictions]


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

    topks_correct = []
    for k in ks:
        top_k_preds = top_max_k_inds[:, :k]  # Get top-k indices

        correct = 0
        for i in range(labels.size(0)):
            # Create a mask to identify the top-k predictions
            top_k_mask = torch.zeros(labels.size(1), dtype=torch.bool)
            top_k_mask[top_k_preds[i]] = True

            # Check if all true labels are within the top-k predictions
            if torch.all(labels[i][top_k_mask]):
                correct += 1

        topks_correct.append(correct)

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


def accuracies_multi_label(preds, labels):
    """
    Compute the accuracies of each label and weighted average value for multi label classification.

    Args:
        preds (tensor):
        labels (tensor):

    Return:
        label_accuracies (list):
        weighted_accuracy (float):
    """

    num_cls = preds.size(1)

    preds = (preds > 0.5).float()

    label_accuracies = []
    total_true_positive = torch.zeros(num_cls)
    total_true = torch.zeros(num_cls)

    for j in range(num_cls):
        true_positive = torch.sum((preds[:, j] == 1) & (labels[:, j] == 1))
        total_true_positives = torch.sum(labels[:, j] == 1)

        accuracy = true_positive / total_true_positives if total_true_positives > 0 else torch.tensor(0.0)
        label_accuracies.append(accuracy.item())

        total_true_positive[j] = true_positive
        total_true[j] = total_true_positives

    total_true_positive_sum = torch.sum(total_true_positive)
    total_true_sum = torch.sum(total_true)
    weighted_accuracy = total_true_positive_sum / total_true_sum if total_true_sum > 0 else torch.tensor(0.0)

    return weighted_accuracy.item() * 100.0


def f1_scores(preds, labels, average='macro'):
    """
    Computes the F1 score for single label classification.

    Args:
        preds (tensor): Predictions with probabilities or scores, shape (N, ClassNum).
        average (str): Averaging method for F1 score. Options are 'micro', 'macro', 'weighted', or 'samples'.

    Returns:
        f1_score (float): F1 score for multi-label classification.
    """
    # Convert tensors to numpy arrays
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    preds = np.argmax(preds, axis=1)

    zero_division = 0

    # Binarize predictions
    f1 = f1_score(labels, preds, average=average, zero_division=zero_division)

    return f1



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
    labels = (labels >= threshold).astype(int)

    zero_division = 0

    # Binarize predictions
    f1 = f1_score(labels, preds, average=average, zero_division=zero_division)

    return f1
