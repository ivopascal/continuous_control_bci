import itertools
import pickle

import mne
import numpy as np
import scipy


def make_rough_emg_events(eeg_stream, emg_stream):
    prediction_timestamps = emg_stream['time_stamps']
    prediction_values = emg_stream['time_series']

    events = np.stack(
        [(prediction_timestamps - eeg_stream['time_stamps'][0]) * 2048, np.zeros(prediction_timestamps.shape),
         prediction_values[:, 0]]).T

    bits = prediction_values
    index = 0
    starting_points = []
    for bit, group in itertools.groupby(bits):
        length = len(list(group))
        if length >= 19:
            starting_points.append(events[index, :])
        index += length

    # So at the start
    starting_points = np.array(starting_points).astype('int32')
    starting_points[:, 0] = starting_points[:, 0] - int(0.2 * 2048)

    return starting_points


def make_precise_emg_events(raw, emg_model_path="./data/pilot_1/emg_model.pkl", interval=0.05, epoch_time=0.2, ):
    with open(emg_model_path, 'rb') as f:
        emg_model = pickle.load(f)

    raw_emg = raw.copy().pick(['emg'])

    filters = [
        mne.filter.create_filter(raw_emg.get_data(), l_freq=30, h_freq=500, method='iir',
                                 phase='forward', sfreq=raw.info['sfreq']),
        mne.filter.create_filter(raw_emg.get_data(), l_freq=51, h_freq=49, method='iir',
                                 phase='forward', sfreq=raw.info['sfreq']),
    ]
    # We do this strange to make is causal filters, in line with the model
    raw_data = scipy.signal.sosfilt(filters[0]['sos'], raw_emg.get_data())
    raw_data = scipy.signal.sosfilt(filters[1]['sos'], raw_data)
    raw_emg = mne.io.RawArray(raw_data, raw_emg.info)

    # Extract samples to classify
    emg_fine_epochs = mne.make_fixed_length_epochs(
        raw_emg,
        duration=epoch_time,
        overlap=epoch_time - interval,
        reject_by_annotation=False,
    )

    # Make predictions and remap
    emg_fine_preds = emg_model.predict(np.abs(emg_fine_epochs.get_data()).mean(axis=2))
    emg_fine_preds[emg_fine_preds == 0] = -1.0
    emg_fine_preds[emg_fine_preds == 2] = 0.0
    emg_fine_preds[emg_fine_preds == 1] = 1.0

    timestamps = np.arange(0, len(raw_emg.times) - epoch_time * 2048, interval * 2048)
    timestamps = timestamps + (epoch_time - interval) * 2048

    all_pred_events = np.stack([timestamps, np.zeros(emg_fine_preds.shape), emg_fine_preds]).T.astype('int32')

    bits = emg_fine_preds
    index = 0
    starting_point_events = []
    for bit, group in itertools.groupby(bits):
        length = len(list(group))
        if length * interval >= 3.75:
            starting_point_events.append(all_pred_events[index, :])
            # print(f"{length  * interval} seconds of {all_pred_events[index, 2]}")
        index += length

    # So at the start
    starting_point_events = np.array(starting_point_events).astype('int32')
    starting_point_events[:, 0] = starting_point_events[:, 0]

    return starting_point_events
