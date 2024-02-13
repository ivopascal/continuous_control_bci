import mne
import numpy as np
from mne.preprocessing import read_ica

from continuous_control_bci.data.load_data import load_calibration
from continuous_control_bci.data.preprocessing import make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier, visualise_csp
from continuous_control_bci.util import SUBJECT_IDS


def main():
    f1s = []
    for subject_id in SUBJECT_IDS:
        raw = load_calibration(subject_id)

        raw = raw.set_eeg_reference(ref_channels=["Cz"], ch_type='eeg')
        raw.drop_channels('Cz')
        raw.filter(l_freq=5, h_freq=35)

        ica = read_ica(f'./data/ica/P{subject_id}-calibration-ica.fif')
        ica.apply(raw)

        include_rest = True
        epochs = make_epochs(raw, include_rest=include_rest)
        X_train, _, y_train, _ = epochs_to_train_test(epochs)
        print("Training classifier. This may take a while..")
        clf, y_pred = create_csp_classifier(X_train, y_train)
        f1 = visualise_csp(subject_id, raw, clf.steps[0][1], y_train, y_pred, include_rest='true', kind='calibration')
        f1s.append(f1)

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()


