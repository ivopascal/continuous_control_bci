import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import read_ica
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from continuous_control_bci.data.load_data import load_calibration
from continuous_control_bci.data.preprocessing import make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier, get_driving_epochs_for_csp
from sklearn.metrics import f1_score

from continuous_control_bci.util import emg_classes_to_eeg_classes, SUBJECT_IDS


def main():
    f1s = []
    for subject_id in SUBJECT_IDS:
        raw = load_calibration(subject_id)
        raw.set_eeg_reference()
        raw.filter(l_freq=5, h_freq=35)

        ica = read_ica(f'./data/ica/P{subject_id}-calibration-ica.fif')
        ica.apply(raw)

        include_rest = True
        epochs = make_epochs(raw, include_rest=include_rest)
        X_train, _, y_train, _ = epochs_to_train_test(epochs)
        print("Training classifier. This may take a while..")
        rank = {
            'eeg': X_train.shape[1] - len(ica.exclude),
            'mag': 32,
        }
        clf, y_pred = create_csp_classifier(X_train, y_train, rank)
        print("Classifier trained!")
        if include_rest:
            target_names = ["Left", "Right", "Rest"]
        else:
            target_names = ["Left", "Right"]

        print(f"Subject {subject_id} calibration")
        print(classification_report(y_train, y_pred, target_names=target_names))

        driving_epochs, _, _ = get_driving_epochs_for_csp(subject_id, include_rest, ica_kind='calibration')
        X_driving = driving_epochs.get_data(copy=True, picks=['eeg'])
        y_driving = emg_classes_to_eeg_classes(driving_epochs.events[:, -1])

        y_driving_pred = clf.predict(X_driving)
        print(f"Transfer performance")
        print(classification_report(y_driving, y_driving_pred, target_names=target_names))

        ConfusionMatrixDisplay.from_predictions(y_driving, y_driving_pred, display_labels=target_names,
                                                normalize='true')
        f1 = f1_score(y_driving, y_driving_pred, average='macro')
        plt.title("Confusion matrix on transfer")
        plt.savefig(f"./figures/{subject_id}_confusion_matrix_transfer.pdf")
        plt.close()
        f1s.append(f1)

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)
    for f1 in f1s:
        print(f1)


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()

