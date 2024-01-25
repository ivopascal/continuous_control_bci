from typing import Tuple

import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline


def create_csp_classifier(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Pipeline, np.ndarray]:
    """
    Trains a CSP classifier on all the data.
    First, however it runs 5-fold cross validation to make cross-validated predictions.
    This the resulting predictions are returns for a fair evaluation, with an optimal model for the training data.
    :param X_train:
    :param y_train:
    :return:
    """
    clf_eeg = Pipeline([("CSP", CSP(n_components=6, reg=None, log=True, norm_trace=False)),
                        ("classifier", LogisticRegression())])
    y_pred = cross_val_predict(clf_eeg, X_train, y_train, cv=5)

    clf_eeg.fit(X_train, y_train)

    return clf_eeg, y_pred
