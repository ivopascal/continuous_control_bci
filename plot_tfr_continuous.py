import matplotlib
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.time_frequency import tfr_multitaper

from continuous_control_bci.data.emg_events import make_rough_emg_events
from continuous_control_bci.data.load_data import load_driving
from continuous_control_bci.util import SUBJECT_IDS
from continuous_control_bci.visualisation.erds import plot_tfr


def main():
    tmin = -3
    tmax = 5
    buffer = 0.5
    time_offset = 1.25
    event_ids = dict(left=-1, right=1)

    all_epochs = []
    for subject_id in SUBJECT_IDS:
        driving_recording = load_driving(subject_id)
        driving_recording.raw.set_eeg_reference()

        driving_recording.raw.filter(l_freq=1, h_freq=35)

        driving_recording.raw = mne.preprocessing.compute_current_source_density(driving_recording.raw)

        matplotlib.use('Agg')
        events = make_rough_emg_events(driving_recording.eeg_stream, driving_recording.emg_stream)

        epochs = mne.Epochs(
            driving_recording.raw,
            events,
            event_ids,
            tmin=tmin - buffer - time_offset,
            tmax=tmax + buffer - time_offset,
            baseline=None,
            preload=True,
        )

        epochs.pick(["C3", "C4"])
        epochs.shift_time(time_offset)
        all_epochs.append(epochs)

        event_ids = dict(left=0, right=1)
        plot_tfr(epochs, subject_id, baseline=(-3, 0), tmin=tmin, tmax=tmax, event_ids=event_ids, kind='driving')
        plot_tfr(epochs, subject_id, baseline=None, tmin=tmin, tmax=tmax, event_ids=event_ids, kind='driving')

    all_epochs = mne.concatenate_epochs(all_epochs)
    plot_tfr(all_epochs, "all_subjects", baseline=(-3, 0), tmin=tmin, tmax=tmax, event_ids=event_ids,  kind='driving')
    plot_tfr(all_epochs, "all_subjects", baseline=None, tmin=tmin, tmax=tmax, event_ids=event_ids, kind='driving')



if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()
