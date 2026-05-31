# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLoss(nn.Module):

    def __init__(self, pos_weight=10.0, reduction='mean', **kwargs):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, x, y):

        pos_weight_tensor = torch.tensor([self.pos_weight], device=x.device, dtype=x.dtype)

        loss = F.binary_cross_entropy_with_logits(
            x, y,
            pos_weight=pos_weight_tensor,
            reduction='none'
        )

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,
                 pos_weight=1.0, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, x, y):

        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)

        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1

        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)

        loss = - one_sided_w * (los_pos * self.pos_weight + los_neg)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([1 - alpha, alpha])
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        at = self.alpha.gather(0, targets.data.view(-1)).reshape(inputs.shape)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def bce_loss(preds, targets):
    device = preds.device
    pos_weight = torch.Tensor([300]).to(device)
    criterion_weighted = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion_weighted(preds, targets)


def calculate_mattews_correlation_coefficient(preds, targets):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    tp = torch.sum(preds * targets)
    tn = torch.sum((1 - preds) * (1 - targets))
    fp = torch.sum(preds * (1 - targets))
    fn = torch.sum((1 - preds) * targets)
    mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc.cpu().item()


# return auc scores
def calculate_auc(targets, preds):
    return roc_auc_score(targets.reshape(-1).cpu().numpy(), preds.reshape(-1).cpu().numpy(), average='micro')


def calculate_auc_fpr_tpr(targets, preds):
    # return auc scores, fpr, tpr
    fpr, tpr, thresholds = roc_curve(targets.reshape(-1).cpu().numpy(), preds.reshape(-1).cpu().numpy())
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr


def evaluate_f1_precision_recall(preds, targets, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(preds) * torch.Tensor(targets))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(preds)).sum()
    true_p = targets.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return precision, recall, f1_score


def rna_evaluation(preds, targets, eps=1e-11):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    tp = torch.sum(preds * targets)
    tn = torch.sum((1 - preds) * (1 - targets))
    fp = torch.sum(preds * (1 - targets))
    fn = torch.sum((1 - preds) * targets)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)  # accuracy
    prec = (tp + eps) / (tp + fp + eps)  # precision
    recall = (tp + eps) / (tp + fn + eps)  # recall
    sens = (tp + eps) / (tp + fn + eps)  # senstivity
    spec = (tn + eps) / (tn + fp + eps)  # spec

    F1 = 2 * ((prec * sens) / (prec + sens))
    MCC = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return accuracy, prec, recall, sens, spec, F1, MCC.cpu().item()


def loglik_nats(model, x):
    """Compute the log-likelihood in nats."""
    return - model.log_prob(x).mean()


def loglik_bpd(model, x, data_fcn_2, data_seq_raw, set_max_len, contact_masks):
    """Compute the log-likelihood in bits per dim."""
    return - model.log_prob(x, data_fcn_2, data_seq_raw, set_max_len, contact_masks).sum() / (
                math.log(2) * x.shape.numel())


def elbo_nats(model, x):
    """
    Compute the ELBO in nats.
    Same as .loglik_nats(), but may improve readability.
    """
    return loglik_nats(model, x)


def elbo_bpd(model, x, data_fcn_2, data_seq_raw, set_max_len, contact_masks):
    """
    Compute the ELBO in bits per dim.
    Same as .loglik_bpd(), but may improve readability.
    """
    return loglik_bpd(model, x, data_fcn_2, data_seq_raw, set_max_len, contact_masks)