from typing import Tuple

import mne.io
import numpy as np
from sklearn.model_selection import train_test_split

LEFT_HAND_EVENT = "769"
RIGHT_HAND_EVENT = "770"
END_OF_TRIAL_EVENT = "800"  # Used for rests


def apply_causal_filters(raw: mne.io.Raw, l_eeg=2, u_eeg=30) -> mne.io.Raw:
    raw = raw.notch_filter(np.arange(50, 501, 50), picks='emg', method='fir',  # noqa
                           phase='minimum')  # Multiple notches only implemented for FIR
    raw = raw.notch_filter(50, method='fir', phase='minimum')
    raw = raw.filter(30, 500, picks='emg', method='iir', phase='forward')
    raw = raw.filter(l_eeg, u_eeg, picks='eeg', method='iir', phase='forward')

    return raw


def make_epochs(raw: mne.io.Raw, include_rest=True) -> mne.Epochs:
    events, _ = mne.events_from_annotations(raw, event_id={LEFT_HAND_EVENT: 1,
                                                           RIGHT_HAND_EVENT: 2,
                                                           END_OF_TRIAL_EVENT: 3
                                                           })
    # Now we'd expect the movement 1 + 1.25 = 2.25 seconds in. The exact timing may be learned
    # It should stay until 6 seconds in. The first and last second should both be "empty"
    # These would be the optimal "during movement" timings. For visualisation, you might also consider
    # preparation or rebound effects.
    tmin = 2.25
    tmax = 2.25 + 3.75

    if include_rest:
        event_ids = dict(left=1, right=2, rest=3)
    else:
        event_ids = dict(left=1, right=2)

    epochs = mne.Epochs(
        raw,
        events,
        event_ids,
        tmin,
        tmax + 0.5,
        baseline=None,
        preload=True,
    )

    return epochs


def epochs_to_train_test(epochs: mne.Epochs, picks=["eeg"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Makes X_train, X_test, y_train, y_test from epochs. X will have shape (Trials, Channels, Time Samples). The
    testing data should not be used for cross validation. You should do k-fold cross validation with the returned
    train data.
    :param picks: Types of channels to select. Should be
    :param epochs: Epochs from which to create training and testing data
    :return: Training and testing data in the form of X_train, X_test, y_train, y_test
    """

    X = epochs.get_data(copy=True, picks=picks)
    y = epochs.events[:, -1] - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test

