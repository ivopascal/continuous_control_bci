from continuous_control_bci.data.load_data import load_calibration, load_driving
from continuous_control_bci.preprocessing.ica import show_ica
from continuous_control_bci.util import SUBJECT_IDS


def main():
    for subject in SUBJECT_IDS:
        calibration = load_calibration(subject)
        show_ica(calibration, subject)

        driving_recording = load_driving(subject)
        show_ica(driving_recording.raw, subject)


if __name__ == "__main__":
    main()
