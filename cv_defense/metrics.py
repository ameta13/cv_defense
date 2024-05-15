from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred, average="macro"):
    return precision_score(y_true, y_pred, average=average)


def recall(y_true, y_pred, average="macro"):
    return recall_score(y_true, y_pred, average=average)


def f1_score(y_true, y_pred, average="macro"):
    return f1_score(y_true, y_pred, average=average)


def clean_accuracy(y_true_clean, y_pred_clean):
    return accuracy_score(y_true_clean, y_pred_clean)


def attack_success_rate(y_true_poisoned, y_pred_poisoned, target_label):
    successful_attacks = np.sum((y_pred_poisoned == target_label) & (y_true_poisoned != target_label))
    total_attacks = np.sum(y_true_poisoned != target_label)
    return successful_attacks / total_attacks


def attack_deduction_rate(y_true_poisoned, y_pred_poisoned_original, y_pred_poisoned_defended, target_label):
    asr_before_defense = attack_success_rate(y_true_poisoned, y_pred_poisoned_original, target_label)
    asr_defended = attack_success_rate(y_true_poisoned, y_pred_poisoned_defended, target_label)
    return asr_before_defense - asr_defended


def robust_accuracy(y_true_poisoned, y_pred_poisoned):
    return accuracy_score(y_true_poisoned, y_pred_poisoned)


def clean_accuracy_drop(y_true_clean, y_pred_clean, y_true_clean_defended, y_pred_clean_defended):
    clean_acc = clean_accuracy(y_true_clean, y_pred_clean)
    clean_acc_defended = clean_accuracy(y_true_clean_defended, y_pred_clean_defended)
    return clean_acc - clean_acc_defended


def clean_classification_confidence(y_pred_clean):
    return np.mean(y_pred_clean.max(axis=1))
