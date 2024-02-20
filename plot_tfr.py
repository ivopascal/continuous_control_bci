import matplotlib
import mne
from mne.preprocessing import read_ica
from tqdm import tqdm

from continuous_control_bci.data.load_data import load_calibration
from continuous_control_bci.data.preprocessing import make_epochs
from continuous_control_bci.util import SUBJECT_IDS
from continuous_control_bci.visualisation.erds import plot_tfr


def main():
    tmin = -3.0
    tmax = 5.0
    buffer = 0.5
    event_ids = dict(left=0, right=1)

    all_epochs = []
    for subject_id in tqdm(SUBJECT_IDS):
        raw = load_calibration(subject_id)
        raw.set_eeg_reference()
        raw.filter(l_freq=1, h_freq=35)

        ica = read_ica(f'./data/ica/P{subject_id}-calibration-ica.fif')
        ica.apply(raw)

        raw = mne.preprocessing.compute_current_source_density(raw)

        matplotlib.use('Agg')

        epochs = make_epochs(raw, tmin=tmin-buffer, tmax=tmax+buffer, include_rest=False)
        epochs.pick(["C3", "C4"])

        plot_tfr(epochs, subject_id, baseline=(-3, 0), tmin=tmin, tmax=tmax, event_ids=event_ids, kind='calibration')
        plot_tfr(epochs, subject_id, baseline=None, tmin=tmin, tmax=tmax, event_ids=event_ids, kind='calibration')
        all_epochs.append(epochs)

    all_epochs = mne.concatenate_epochs(all_epochs)
    plot_tfr(all_epochs, "all_subjects", baseline=(-3, 0), tmin=tmin, tmax=tmax, event_ids=event_ids, kind='calibration')
    plot_tfr(all_epochs, "all_subjects", baseline=None, tmin=tmin, tmax=tmax, event_ids=event_ids, kind='calibration')


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()
