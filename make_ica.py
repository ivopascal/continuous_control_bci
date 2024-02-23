from tqdm import tqdm

from continuous_control_bci.data.load_data import load_calibration, load_driving
from continuous_control_bci.preprocessing.ica import run_and_save_ica
from continuous_control_bci.util import SUBJECT_IDS


def main():
    for subject in tqdm(SUBJECT_IDS):
        calibration = load_calibration(subject)
        run_and_save_ica(calibration, subject, kind='calibration')

        driving_recording = load_driving(subject)
        run_and_save_ica(driving_recording.raw, subject, kind='driving')


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()
