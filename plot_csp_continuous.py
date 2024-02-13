import itertools
import pickle

import mne
import numpy as np
import pyxdf
import scipy
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.inspection import permutation_importance

from continuous_control_bci.data.emg_events import make_rough_emg_events
from continuous_control_bci.data.preprocessing import apply_causal_filters, \
    epochs_to_train_test, manual_clean_ica
from continuous_control_bci.dataclass import DrivingStreams
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier
from continuous_control_bci.util import channel_names, emg_classes_to_eeg_classes, SUBJECT_IDS

EEG_MAPPING = {name: ch_type for name, ch_type in zip(channel_names[:-8], ["eeg"] * len(channel_names[:-8]))}
EMG_MAPPING = {name: ch_type for name, ch_type in zip(channel_names[-8:-4], ["emg"] * 4)}
EOG_MAPPING = {name: ch_type for name, ch_type in zip(channel_names[-4:], ["eog"] * 4)}

CHANNEL_TYPE_MAPPING = {
    **EEG_MAPPING,
    **EMG_MAPPING,
    **EOG_MAPPING,
}


def get_streams_from_xdf(fname) -> DrivingStreams:
    streams, header = pyxdf.load_xdf(fname)

    eeg_stream = None
    emg_stream = None
    unity_stream = None
    position_stream = None
    for stream in streams:
        stream_name = stream['info']['name'][0]
        if stream_name == "BioSemi":
            eeg_stream = stream
        if stream_name == 'PredictionStream':
            emg_stream = stream
        if stream_name == "UnityStream":
            unity_stream = stream
        if stream_name == "UnityPositionStream":
            position_stream = stream

    raw = mne.io.RawArray(eeg_stream['time_series'].T[1:41, :] / 10e5, info=mne.create_info(channel_names, sfreq=2048))

    return DrivingStreams(raw, eeg_stream, emg_stream, unity_stream, position_stream)


def main():
    fname = f"./data/sub-P{subject_id}/ses-S001/eeg/sub-P{subject_id}_ses-S001_task-Default_run-001_eeg.xdf"
    raw, eeg_stream, emg_stream, _ = get_streams_from_xdf(fname)
    raw.set_channel_types(CHANNEL_TYPE_MAPPING)
    raw.set_montage("biosemi32", on_missing='raise')
    raw.set_eeg_reference()

    raw.set_eeg_reference(ref_channels=["Cz"], ch_type='eeg')
    raw.drop_channels('Cz')
    # raw = apply_causal_filters(raw)
    # raw.set_eeg_reference()
    raw = raw.filter(5, 35, picks='eeg')
    # events = make_precise_emg_events(raw)
    events = make_rough_emg_events(eeg_stream, emg_stream)

    event_ids = dict(left=-1, rest=0, right=1)
    epochs = mne.Epochs(
        raw,
        events,
        event_ids,
        tmin=0.5,
        tmax=3.25,
        baseline=None,
        preload=True,
        picks='eeg',
    )

    X_train, _, y_train, _ = epochs_to_train_test(epochs)
    print("Training classifier. This may take a while..")
    clf, y_pred = create_csp_classifier(X_train, y_train)
    print("Classifier trained!")
    target_names = ["Left", "Right", "Rest"]

    print(f"Subject {subject_id}")
    print(classification_report(emg_classes_to_eeg_classes(y_train), emg_classes_to_eeg_classes(y_pred),
                                target_names=target_names))
    ConfusionMatrixDisplay.from_predictions(emg_classes_to_eeg_classes(y_train), emg_classes_to_eeg_classes(y_pred),
                                            display_labels=target_names, normalize='true')
    plt.title("Confusion matrix on driving data")
    plt.savefig(f"./figures/{subject_id}_confusion_matrix_driving.pdf")
    plt.close()

    csp = clf.steps[0][1]
    csp.plot_filters(info=raw.pick('eeg').info, show_names=True, sphere='eeglab', colorbar=False)
    model_fi = permutation_importance(clf.steps[1][1], csp.transform(X_train), y_train)
    print(f"Feature importances {model_fi['importances_mean']}")
    plt.savefig(f"./figures/{subject_id}_csp_filters_continuous.pdf")
    plt.close()

    f1 = f1_score(y_train, y_pred, average='macro')
    return f1


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa

    f1s = []
    for subject_id in SUBJECT_IDS:
        f1s.append(main())

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)

