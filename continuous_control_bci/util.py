channel_names = [
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
    "Left-extension",
    "Left-flexion",
    "Right-flexion",
    "Right-extension",
    "LHEOG",
    "RHEOG",
    "UVEOG",
    "LVEOG",
]

SUBJECT_IDS = ["061",
               "066",
               "126",
               "280",
               "302",
               "336",
               "381",
               "406",
               "530",
               "587",
               "643",
               "656",
               "682",
               "744",
               "812",
               "840",
               "854",
               # "942",
               "968",
               "986"]

EEG_MAPPING = {name: ch_type for name, ch_type in zip(channel_names[:-8], ["eeg"] * len(channel_names[:-8]))}
EMG_MAPPING = {name: ch_type for name, ch_type in zip(channel_names[-8:-4], ["emg"] * 4)}
EOG_MAPPING = {name: ch_type for name, ch_type in zip(channel_names[-4:], ["eog"] * 4)}

CHANNEL_TYPE_MAPPING = {
    **EEG_MAPPING,
    **EMG_MAPPING,
    **EOG_MAPPING,
}


def emg_classes_to_eeg_classes(classes):
    # Translates predictions encoded as -1, 0, 1 for left, rest, right
    # Into 0, 1, 2 for left, right, rest
    classes = classes.copy()
    classes[classes == 0] = 2
    classes[classes == -1] = 0
    classes[classes == 1] = 1

    return classes
