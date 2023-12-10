import numpy as np
import sklearn.metrics

def calculate_eer(y_true: np.array, y_score: np.array):
    fpr, tpr, t = sklearn.metrics.roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_t = t[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_1 + eer_2) / 2
    return eer