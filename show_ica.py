import matplotlib
from tqdm import tqdm
import mne

from continuous_control_bci.data.emg_events import make_rough_emg_events
from continuous_control_bci.data.load_data import load_calibration, load_driving
from continuous_control_bci.data.preprocessing import make_epochs
from continuous_control_bci.preprocessing.ica import show_ica
from continuous_control_bci.util import SUBJECT_IDS


def main():
    for subject in tqdm(SUBJECT_IDS):
        calibration = load_calibration(subject)
        calibration.filter(0.2, 40)

        calibration_epochs = make_epochs(calibration, include_rest=False)
        show_ica(calibration_epochs, subject, kind='calibration', filter=False)

        driving_recording = load_driving(subject)

        events = make_rough_emg_events(driving_recording.eeg_stream, driving_recording.emg_prediction_stream)
        driving_recording.raw.filter(0.2, 40)
        driving_epochs = mne.Epochs(
            driving_recording.raw,
            events,
            event_id=dict(left=-1, right=1),
            tmin=0.5,
            tmax=3.25,
            baseline=None,
            preload=True,
        )
        show_ica(driving_epochs, subject, kind='driving', filter=False)


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    mne.viz.use_browser_backend('matplotlib')
    matplotlib.use('QtAgg')
    main()
