import os.path

from matplotlib import pyplot as plt
from mne.preprocessing import ICA, read_ica


def run_and_save_ica(raw, subject_id, overwrite=False, kind='calibration'):
    save_path = f'./data/ica/P{subject_id}-{kind}-ica.fif'
    if os.path.exists(save_path) and not overwrite:
        return

    raw_filtered = raw.copy().filter(1, 30)
    ica = ICA(n_components=32, random_state=42)
    ica.fit(raw_filtered)
    ica.save(save_path, overwrite=overwrite)


def show_ica(raw, subject_id, kind='calibration', overwrite=True, filter=True):
    ica = read_ica(f'./data/ica/P{subject_id}-{kind}-ica.fif')
    print(f"{kind} subject {subject_id}")

    if filter:
        raw = raw.copy().filter(0.2, 40)
    ica.plot_components(inst=raw)
    plt.show()
    if overwrite:
        ica.save(f'./data/ica/P{subject_id}-{kind}-ica.fif', overwrite=True)

