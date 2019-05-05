import os
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from sklearn.utils.class_weight import compute_class_weight


def EER(labels, scores):
    """
    Computes EER (and threshold at which EER occurs) given a list of (gold standard) True/False labels
    and the estimated similarity scores by the verification system (larger values indicates more similar)
    Sources: https://yangcha.github.io/EER-ROC/ & https://stackoverflow.com/a/49555212/1493011
    """
    weights = compute_class_weight('balanced', [0, 1], labels)
    weights = weights[1] * labels + weights[0] * (1 - labels)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, sample_weight=weights, drop_intermediate=False)
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)



    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    thresh = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer, thresh


def threshold_select(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=True)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, thresholds[fpr.index(0.0)]
