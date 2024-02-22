from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import mne.io
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from sklearn.model_selection import train_test_split

from continuous_control_bci.dataclass import DrivingStreams

LEFT_HAND_EVENT = "769"
RIGHT_HAND_EVENT = "770"
END_OF_TRIAL_EVENT = "800"  # Used for rests


def apply_causal_filters(raw: mne.io.Raw, l_eeg=5, u_eeg=35) -> mne.io.Raw:
    raw = raw.notch_filter(np.arange(50, 501, 50), picks='emg', method='fir',  # noqa
                           phase='minimum')  # Multiple notches only implemented for FIR
    # raw = raw.notch_filter(50, method='fir', phase='minimum')
    raw = raw.filter(30, 500, picks='emg', method='iir', phase='forward')
    raw = raw.filter(l_eeg, u_eeg, picks='eeg', method='iir', phase='forward')

    return raw


def make_epochs(raw: mne.io.Raw, include_rest=True, tmin=1.25, tmax=5) -> mne.Epochs:
    events, _ = mne.events_from_annotations(raw, event_id={LEFT_HAND_EVENT: 0,
                                                           RIGHT_HAND_EVENT: 1})
    # Now we'd expect the movement 1.25 = 1.25 seconds in. The exact timing may be learned
    # It should stay for 3.75 seconds. The first and last second should both be "empty"
    # These would be the optimal "during movement" timings. For visualisation, you might also consider
    # preparation or rebound effects.

    event_ids = dict(left=0, right=1)

    ch_type = raw.get_channel_types()[0]
    if ch_type == "eeg":
        reject = {"eeg": 100e-6}
    elif ch_type == "csd":
        reject = {"csd": 100e-3}
    else:
        raise ValueError

    epochs = mne.Epochs(
        raw,
        events,
        event_ids,
        tmin,
        tmax,
        reject=reject,
        baseline=None,
        preload=True,
    )

    if include_rest:
        # For rest we want to have 0.25s buffer after end of movement
        # But we do want to keep the same time span
        # This is okay because after END_OF_TRIAL there is
        # [1.5, 3.5] seconds of black screen
        # 3 seconds of cross
        # This means tmax cannot be larger than 4.5
        # We take tmin = 0.5, so tmax ends at 4.25
        tmin = 0.5
        tmax = 0.5 + 3.75
        assert tmax < 4.5
        events, _ = mne.events_from_annotations(raw, event_id={
            END_OF_TRIAL_EVENT: 2
        })
        rest_epochs = mne.Epochs(
            raw,
            events,
            dict(rest=2),
            tmin,
            tmax,
            reject=reject,
            baseline=None,
            preload=True,
        )
        rest_epochs.shift_time(1.25 - 0.5)  # This only changes the metadata time. The data stays the same

        epochs = mne.concatenate_epochs([epochs, rest_epochs], add_offset=False)

    return epochs


def epochs_to_train_test(epochs: mne.Epochs, picks=["eeg"]) -> Tuple[np.ndarray, None, np.ndarray, None]:
    """
    Makes X_train, X_test, y_train, y_test from epochs. X will have shape (Trials, Channels, Time Samples). The
    testing data should not be used for cross validation. You should do k-fold cross validation with the returned
    train data.
    :param picks: Types of channels to select. Should be
    :param epochs: Epochs from which to create training and testing data
    :return: Training and testing data in the form of X_train, X_test, y_train, y_test
    """

    X = epochs.get_data(copy=True, picks=picks)
    y = epochs.events[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)

    return X, None, y, None


CALIBRATION_MRCP_BAD_ICS = {
    "066": [1, 4, 13, 0],
    "280": [1, 0, 5, 11],
    "302": [0, 1, 14],
    "406": [0, 2, 5, 1],
    "530": [0, 1, 2],
    "587": [0, 2, 6, 11, 17],
    "643": [0, 3, 1, 5, 9],
    "682": [0, 1, 3],
    "744": [0, 1, 3, 7, 11],
    "812": [1, 3, 6, 2],
    "840": [0, 3, 6],
    "854": [1, 2, 9],
    "942": [0, 1, 3, 5, 8, 13, 15],
    "968": [0, 1, 4, 9, 3],
    "986": [1, 0, 10]
}

DRIVING_MRCP_BAD_ICS = {
    "066": [0, 1, 2, 3, 6, 5, 8, 9, 7, 4],
    "280": [0, 1, 2, 3, 5, 18],
    "302": [1, 2, 4, 3, 0, 6],
    "406": [1, 2, 10],
    "530": [0, 1, 2, 4, 5, 7, 10],
    "587": [2, 3],
    "643": [0, 1, 10],
    "682": [0, 1, 2, 12],
    "744": [0, 1, 2, 3],
    "812": [2, 3, 17],
    "840": [0, 1, 2, 5],
    "854": [0, 1, 2],
    "942": [0, 2, 4, 8, 13],
    "968": [0, 4],
    "986": [1, 0],
}


def manual_clean_ica(raw: mne.io.Raw, exclude=None, subject_id=None, ic_dict=None):
    matplotlib.use("MacOSX")
    ica = ICA(n_components=20, random_state=42)
    ica.fit(raw)

    if subject_id and ic_dict and subject_id in ic_dict.keys():
        ica.exclude = ic_dict[subject_id]
        print(f"Rejecting ICs: {ica.exclude}")
        ica.apply(raw)
        return

    if exclude:
        ica.exclude = exclude
        print(f"Rejecting ICs: {ica.exclude}")
        ica.apply(raw)
        return

    bad_eog, _ = ica.find_bads_eog(raw)
    print(subject_id)
    print(f"Bad EOG predicted: {bad_eog}")
    ica.plot_components()
    plt.show()
    ica.plot_sources(raw)
    plt.show()
    print(subject_id)
    print(f"Rejecting ICs: \"{subject_id}\": {ica.exclude}")
    ica.apply(raw)


def auto_clean_ica(raw):
    # Incoming raw should be CAR and ideally filtered between 1-100Hz
    ica = ICA(n_components=20, random_state=42, method='infomax', fit_params=dict(extended=True))
    ica.fit(raw)
    # bad_eog, _ = ica.find_bads_eog(raw)
    # print(f"Bad EOG predicted: {bad_eog}")
    # ica.exclude = bad_eog
    iclabel_label_components(raw, ica)
    print(ica.labels_)
    ica.exclude += ica.labels_['muscle']
    ica.exclude += ica.labels_['eog']
    ica.exclude += ica.labels_['ecg']
    ica.exclude += ica.labels_['line_noise']
    ica.exclude += ica.labels_['ch_noise']

    ica.apply(raw)


def remove_driving_rests(driving_streams: DrivingStreams) -> mne.io.RawArray:
    driving_start_marker = 'Lap1Completed'
    driving_end_marker = 'Lap6Completed'

    onsets = []
    durations = []
    offset = driving_streams.eeg_stream['time_stamps'][0]
    start_t = offset  # we reject the start
    onsets.append(0)
    previous_item = driving_end_marker
    for idx, item in enumerate(driving_streams.unity_stream['time_series']):
        if item[0] == driving_start_marker:
            end_t = driving_streams.unity_stream['time_stamps'][idx]

            if previous_item == driving_end_marker:
                durations.append(end_t - start_t)
            elif previous_item == driving_start_marker:
                durations[-1] = end_t - start_t
            else:
                raise ValueError(f"Did not expect previous marker {previous_item}")
            previous_item = item[0]

        if item[0] == driving_end_marker:
            if previous_item == driving_start_marker:
                start_t = driving_streams.unity_stream['time_stamps'][idx]
                onsets.append(start_t - offset)
            previous_item = item[0]

    if previous_item == driving_end_marker:
        end_t = driving_streams.eeg_stream['time_stamps'][-1]
        durations.append(end_t - start_t -1)
    driving_streams.raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=["BAD_break"] * len(onsets)))
    return driving_streams.raw
