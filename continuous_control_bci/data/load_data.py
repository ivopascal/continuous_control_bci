from glob import glob

import mne
from pyxdf import pyxdf

from continuous_control_bci.data.emg_events import make_rough_emg_events
from continuous_control_bci.data.preprocessing import remove_driving_rests
from continuous_control_bci.dataclass import DrivingStreams
from continuous_control_bci.util import channel_names, CHANNEL_TYPE_MAPPING


def load_calibration(subject_id):
    raw = load_from_file(glob(f'./data/sub-P{subject_id}/motor-imagery-csp-{subject_id}-acquisition*.gdf')[0])

    raw = adjust_info(raw)
    return raw


def load_driving(subject_id):
    fname = f"./data/sub-P{subject_id}/ses-S001/eeg/sub-P{subject_id}_ses-S001_task-Default_run-001_eeg.xdf"
    streams = get_streams_from_xdf(fname)
    streams.raw.set_channel_types(CHANNEL_TYPE_MAPPING)
    streams.raw.set_montage("biosemi32", on_missing='raise')

    streams.raw = remove_driving_rests(streams)

    return streams


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

