import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import read_ica
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tqdm import tqdm

from continuous_control_bci.data.load_data import load_calibration
from continuous_control_bci.data.preprocessing import make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier, get_driving_epochs_for_csp
from sklearn.metrics import f1_score

from continuous_control_bci.util import emg_classes_to_eeg_classes, SUBJECT_IDS


def predict_against_threshold(clf, X, t):
    y_driving_pred_prob = clf.predict_proba(X)
    rest = y_driving_pred_prob[:, 2] > t
    y_driving_pred = np.array([2] * len(rest))
    y_driving_pred[~rest] = y_driving_pred_prob[~rest, :2].argmax(axis=1)
    return y_driving_pred


def main():
    f1s = []
    all_y_true = []
    all_y_pred = []

    include_rest = True
    if include_rest:
        target_names = ["Left", "Right", "Rest"]
    else:
        target_names = ["Left", "Right"]

    for subject_id in tqdm(SUBJECT_IDS):
        raw = load_calibration(subject_id)
        raw.set_eeg_reference()
        raw.filter(l_freq=5, h_freq=35)

        ica = read_ica(f'./data/ica/P{subject_id}-calibration-ica.fif')
        ica.apply(raw)

        epochs = make_epochs(raw, include_rest=include_rest)
        X_train, _, y_train, _ = epochs_to_train_test(epochs)
        print("Training classifier. This may take a while..")
        rank = {
            'eeg': X_train.shape[1] - len(ica.exclude),
            'mag': 32,
        }
        clf, y_pred = create_csp_classifier(X_train, y_train, rank)
        print("Classifier trained!")

        # print(f"Subject {subject_id} calibration")
        # print(classification_report(y_train, y_pred, target_names=target_names))

        driving_epochs, _, _ = get_driving_epochs_for_csp(subject_id, include_rest, ica_kind='calibration')
        X_driving = driving_epochs.get_data(copy=True, picks=['eeg'])
        y_driving = emg_classes_to_eeg_classes(driving_epochs.events[:, -1])

        # We're using 10% of the driving data to find the optimal threshold
        threshold_samples = int(X_driving.shape[0] * 0.1)
        ts = np.linspace(0, 1, 30)
        t_scores = [f1_score(y_driving[:threshold_samples], predict_against_threshold(clf, X_driving[:threshold_samples], t), average='micro') for t in ts]
        best_t = ts[np.argmax(t_scores)]

        y_driving_pred = predict_against_threshold(clf, X_driving[threshold_samples:], best_t)

        print(f"Transfer performance")
        print(classification_report(y_driving[threshold_samples:], y_driving_pred, target_names=target_names))

        ConfusionMatrixDisplay.from_predictions(y_driving[threshold_samples:], y_driving_pred, display_labels=target_names,
                                                normalize='true')
        f1 = f1_score(y_driving[threshold_samples:], y_driving_pred, average='micro')
        plt.title("Confusion matrix on transfer")
        plt.savefig(f"./figures/{subject_id}_confusion_matrix_transfer.pdf")
        plt.close()
        f1s.append(f1)
        all_y_true.append(y_driving[threshold_samples:])
        all_y_pred.append(y_driving_pred)

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)
    for f1 in f1s:
        print(f1)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    print(classification_report(all_y_true, all_y_pred, target_names=target_names))
    ConfusionMatrixDisplay.from_predictions(all_y_true, all_y_pred, display_labels=target_names,
                                            normalize='true')
    plt.title(f"Confusion matrix on transfer data")
    plt.savefig(f"./figures/all_subjects_confusion_matrix_transfer.pdf")
    plt.close()


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()

