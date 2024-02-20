import mne
import numpy as np
from tqdm import tqdm

from continuous_control_bci.modelling.csp_classifier import create_csp_classifier, visualise_csp, \
    get_driving_epochs_for_csp
from continuous_control_bci.util import emg_classes_to_eeg_classes, SUBJECT_IDS


def main():
    f1s = []
    all_epochs = []
    include_rest = True
    for subject_id in tqdm(SUBJECT_IDS):
        epochs, driving_recording, ica = get_driving_epochs_for_csp(subject_id, include_rest=include_rest,
                                                                    ica_kind='driving')
        all_epochs.append(epochs)
        X = epochs.get_data(copy=True, picks=['eeg'])
        y_true = emg_classes_to_eeg_classes(epochs.events[:, -1])

        rank = {
            'eeg': X.shape[1] - len(ica.exclude),
            'mag': 32,
        }

        print("Training classifier. This may take a while..")
        clf, y_pred = create_csp_classifier(X, y_true, rank)
        print("Classifier trained!")

        f1 = visualise_csp(subject_id, driving_recording.raw, clf.steps[0][1], y_true,
                           y_pred, include_rest=include_rest, kind='driving')
        f1s.append(f1)

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)

    all_epochs = mne.concatenate_epochs(all_epochs)
    X = all_epochs.get_data(copy=True, picks=['eeg'])
    y_true = emg_classes_to_eeg_classes(all_epochs.events[:, -1])
    print("Training classifier. This may take a while..")
    rank = {
        'eeg': X.shape[1],
        'mag': 32
    }
    clf, y_pred = create_csp_classifier(X, y_true, rank)
    f1 = visualise_csp("all_subjects", driving_recording.raw, clf.steps[0][1], y_true, y_pred, include_rest=include_rest, kind='driving')
    print(f"Global F1 {f1}")


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()
