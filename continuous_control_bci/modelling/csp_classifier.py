from typing import Tuple

import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.decoding import CSP
from mne.preprocessing import read_ica
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from continuous_control_bci.data.emg_events import make_rough_emg_events
from continuous_control_bci.data.load_data import load_driving


def create_csp_classifier(X_train: np.ndarray, y_train: np.ndarray, rank) -> Tuple[Pipeline, np.ndarray]:
    """
    Trains a CSP classifier on all the data.
    First, however it runs 5-fold cross validation to make cross-validated predictions.
    This the resulting predictions are returns for a fair evaluation, with an optimal model for the training data.
    :param X_train:
    :param y_train:
    :return:
    """
    clf_eeg = Pipeline([
        ("CSP", CSP(n_components=6, reg='shrinkage', log=True, rank=rank)),
        ("classifier", LogisticRegression())])
    y_pred = cross_val_predict(clf_eeg, X_train, y_train, cv=5)

    clf_eeg.fit(X_train, y_train)

    return clf_eeg, y_pred


def visualise_csp(subject_id, raw, csp, y_train, y_pred, include_rest=True, kind='calibration'):
    print("Classifier trained!")
    if include_rest:
        target_names = ["Left", "Right", "Rest"]
    else:
        target_names = ["Left", "Right"]

    print(f"Subject {subject_id}")
    print(classification_report(y_train, y_pred, target_names=target_names))
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, display_labels=target_names,
                                            normalize='true')
    f1 = f1_score(y_train, y_pred, average='macro')
    plt.title(f"Confusion matrix on {kind} data")
    plt.savefig(f"./figures/{subject_id}_confusion_matrix_{kind}.pdf")
    plt.close()

    csp.plot_filters(info=raw.pick('eeg').info, show_names=True, sphere='eeglab', colorbar=False)
    plt.savefig(f"./figures/{subject_id}_csp_filters_{kind}.pdf")
    plt.close()

    return f1


def get_driving_epochs_for_csp(subject_id, include_rest=True, ica_kind='driving'):
    driving_recording = load_driving(subject_id)

    driving_recording.raw.set_eeg_reference()
    driving_recording.raw.filter(5, 35, picks='eeg')

    ica = read_ica(f'./data/ica/P{subject_id}-{ica_kind}-ica.fif')
    ica.apply(driving_recording.raw)


    events = make_rough_emg_events(driving_recording.eeg_stream, driving_recording.emg_prediction_stream)

    if include_rest:
        event_ids = dict(left=-1, rest=0, right=1)
    else:
        event_ids = dict(left=-1, right=1)
    epochs = mne.Epochs(
        driving_recording.raw,
        events,
        event_ids,
        tmin=0.5,
        tmax=3.25,
        baseline=None,
        preload=True,
        picks='eeg',
    )

    return epochs, driving_recording, ica
