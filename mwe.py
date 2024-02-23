from glob import glob

import mne
import numpy as np
from mne.preprocessing import read_ica
from tqdm import tqdm

from continuous_control_bci.data.load_data import load_calibration
from continuous_control_bci.data.preprocessing import make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier, visualise_csp
from continuous_control_bci.util import SUBJECT_IDS


def main():
    subject_id = '123'

    raw = mne.io.read_raw_gdf(glob(f'./data/sub-P{subject_id}/motor-imagery-csp-{subject_id}-acquisition*.gdf')[0], preload=True)

    original_channel_names = [f"Channel {i + 1}" for i in range(32)] + [f"EX {i + 1}" for i in range(8)]
    renaming = {original: new for original, new in zip(original_channel_names, channel_names)}
    raw = raw.set_channel_types(dict.fromkeys(["EX 1", "EX 2", "EX 3", "EX 4"], "emg"))
    raw = raw.set_channel_types(dict.fromkeys(["EX 5", "EX 6", "EX 7", "EX 8"], "eog"))
    raw = raw.rename_channels(renaming)
    raw = raw.set_montage("biosemi32", on_missing='raise')

    raw = raw.set_eeg_reference()

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



if __name__ == "__main__":
    # mne.set_log_level('warning') # noqa
    main()


