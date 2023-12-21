import mne

from continuous_control_bci.util import channel_names


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

