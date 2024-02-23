import mne
import numpy as np
from mne.preprocessing import read_ica
from tqdm import tqdm

from continuous_control_bci.data.load_data import load_calibration
from continuous_control_bci.data.preprocessing import make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier, visualise_csp
from continuous_control_bci.util import SUBJECT_IDS


def main():
    f1s = []
    all_epochs = []
    include_rest = True

    for subject_id in tqdm(SUBJECT_IDS):
        raw = load_calibration(subject_id)

        raw = raw.set_eeg_reference()

        raw.filter(l_freq=5, h_freq=35)

        ica = read_ica(f'./data/ica/P{subject_id}-calibration-ica.fif')
        ica.apply(raw)

        epochs = make_epochs(raw, include_rest=include_rest)
        all_epochs.append(epochs)
        X_train, _, y_train, _ = epochs_to_train_test(epochs)
        X_train *= 1e6
        print("Training classifier. This may take a while..")
        rank = {
            'eeg': X_train.shape[1] - len(ica.exclude),
            'mag': 32,
        }
        clf, y_pred = create_csp_classifier(X_train, y_train, rank)
        f1 = visualise_csp(subject_id, raw, clf.steps[0][1], y_train, y_pred, include_rest=include_rest, kind='calibration')
        f1s.append(f1)

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)

    all_epochs = mne.concatenate_epochs(all_epochs)
    X_train, _, y_train, _ = epochs_to_train_test(all_epochs)
    X_train *= 1e6
    print("Training classifier. This may take a while..")
    rank = {
        'eeg': X_train.shape[1],
        'mag': 32
    }
    clf, y_pred = create_csp_classifier(X_train, y_train, rank)
    f1 = visualise_csp("all_subjects", raw, clf.steps[0][1], y_train, y_pred, include_rest=include_rest, kind='calibration')
    print(f"Global F1 {f1}")
    for i in f1s:
        print(i)


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()


