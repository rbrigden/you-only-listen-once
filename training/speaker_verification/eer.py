import os
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def EER(labels, scores):
    """
    Computes EER (and threshold at which EER occurs) given a list of (gold standard) True/False labels
    and the estimated similarity scores by the verification system (larger values indicates more similar)
    Sources: https://yangcha.github.io/EER-ROC/ & https://stackoverflow.com/a/49555212/1493011
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=True)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

