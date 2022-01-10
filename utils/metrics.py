from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from statistics import mean


def get_mean_sensitivity_specificity(y_true: list, y_pred: list):
    cm_all = confusion_matrix(y_true, y_pred)
    cr_all = classification_report(y_true, y_pred, output_dict=True)

    specificities = []
    for i in range(cm_all.shape[0]):
        total = np.sum(cm_all)
        tp = cm_all[i][i]
        fn = np.sum(cm_all[i, :]) - tp
        fp = np.sum(cm_all[:, i]) - tp
        tn = total - (tp + fp + fn)
        specificity = tn / (tn + fp)
        specificities.append(specificity)

    return cr_all["macro avg"]["recall"], mean(specificities)
