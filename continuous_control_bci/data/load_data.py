import mne

from continuous_control_bci.data.emg_events import make_rough_emg_events
from continuous_control_bci.util import channel_names
from plot_csp_continuous import get_streams_from_xdf, CHANNEL_TYPE_MAPPING


def load_from_file(filename: str):
    return mne.io.read_raw_gdf(filename,
                               preload=True)


def adjust_info(raw: mne.io.Raw) -> mne.io.Raw:
    original_channel_names = [f"Channel {i + 1}" for i in range(32)] + [f"EX {i + 1}" for i in range(8)]
    renaming = {original: new for original, new in zip(original_channel_names, channel_names)}
    raw = raw.set_channel_types(dict.fromkeys(["EX 1", "EX 2", "EX 3", "EX 4"], "emg"))
    raw = raw.set_channel_types(dict.fromkeys(["EX 5", "EX 6", "EX 7", "EX 8"], "eog"))
    raw = raw.rename_channels(renaming)
    raw = raw.set_montage("biosemi32", on_missing='raise')

    return raw


def get_driving_epochs_for_csp(subject_id, include_rest=True):
    fname = f"./data/sub-P{subject_id}/ses-S001/eeg/sub-P{subject_id}_ses-S001_task-Default_run-001_eeg.xdf"
    raw, eeg_stream, emg_stream = get_streams_from_xdf(fname)
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

    if include_rest:
        event_ids = dict(left=-1, rest=0, right=1)
    else:
        event_ids = dict(left=-1, right=1)
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

    return epochs
