import mne
import numpy as np

from continuous_control_bci.modelling.csp_classifier import create_csp_classifier, visualise_csp, \
    get_driving_epochs_for_csp
from continuous_control_bci.util import emg_classes_to_eeg_classes, SUBJECT_IDS


def main():
    f1s = []
    for subject_id in SUBJECT_IDS:
        epochs, driving_recording = get_driving_epochs_for_csp(subject_id)
        X = epochs.get_data(copy=True, picks=['eeg'])
        y_true = emg_classes_to_eeg_classes(epochs.events[:, -1])
        print("Training classifier. This may take a while..")
        clf, y_pred = create_csp_classifier(X, y_true)
        print("Classifier trained!")

        f1 = visualise_csp(subject_id, driving_recording.raw, clf.steps[0][1], emg_classes_to_eeg_classes(y_true),
                           emg_classes_to_eeg_classes(y_pred), include_rest='true', kind='driving')
        f1s.append(f1)

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()
