import matplotlib
import mne
from matplotlib import pyplot as plt
from mne.preprocessing import read_ica
from tqdm import tqdm

from continuous_control_bci.data.emg_events import make_rough_emg_events
from continuous_control_bci.data.load_data import load_driving
from continuous_control_bci.util import SUBJECT_IDS


def main():
    driving_recording = load_driving(subject_id)
    driving_recording.raw.set_eeg_reference()

    matplotlib.use('Agg')

    driving_recording.raw.filter(0.1, 3, picks='eeg', method='iir', phase='forward')

    ica = read_ica(f'./data/ica/P{subject_id}-driving-ica.fif')
    ica.apply(driving_recording.raw)

    events = make_rough_emg_events(driving_recording.eeg_stream, driving_recording.emg_prediction_stream)

    event_ids = dict(left=-1, right=1)

    ch_type = driving_recording.raw.get_channel_types()[0]
    if ch_type == "eeg":
        reject = {"eeg": 100e-6}
    elif ch_type == "csd":
        reject = {"csd": 100e-3}
    else:
        raise ValueError
    epochs = mne.Epochs(
        driving_recording.raw,
        events,
        event_ids,
        tmin=-0.9 - 1.25,
        tmax=5 - 1.25,
        baseline=None,
        reject=reject,
        preload=True,
        picks=ch_type,
    )
    epochs.shift_time(1.25)

    evokeds = dict(
        left=list(epochs["left"].iter_evoked()),
        right=list(epochs["right"].iter_evoked()),
    )

    print(subject_id)
    ylim = [-10, 10]

    mne.viz.plot_compare_evokeds(evokeds, axes='topo', ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[0.0, 1.25])

    plt.savefig(f"./figures/{subject_id}_driving_MRCP.pdf")
    plt.close()

    mne.viz.plot_compare_evokeds(evokeds, picks=['C3'], ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[0.0, 1.25])
    plt.savefig(f"./figures/{subject_id}_driving_MRCP_C3.pdf")
    plt.close()

    mne.viz.plot_compare_evokeds(evokeds, picks=['C4'], ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[0.0, 1.25])
    plt.savefig(f"./figures/{subject_id}_driving_MRCP_C4.pdf")
    plt.close()

    return evokeds


if __name__ == "__main__":
    mne.set_log_level('warning')  # noqa

    all_evokeds = []
    for subject_id in tqdm(SUBJECT_IDS):
        all_evokeds.append(main())

    all_left = []
    all_right = []
    for evokeds in all_evokeds:
        all_left.extend(evokeds['left'])
        all_right.extend(evokeds['right'])

    evokeds = dict(left=all_left, right=all_right)

    ylim = [-5, 5]
    mne.viz.plot_compare_evokeds(evokeds, axes='topo', ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[1.25])

    plt.savefig(f"./figures/all_subjects_driving_MRCP.pdf")
    plt.close()

    mne.viz.plot_compare_evokeds(evokeds, picks=['C3'], ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[1.25])
    plt.savefig(f"./figures/all_subjects_driving_MRCP_C3.pdf")
    plt.close()

    mne.viz.plot_compare_evokeds(evokeds, picks=['C4'], ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[1.25])
    plt.savefig(f"./figures/all_subjects_driving_MRCP_C4.pdf")
    plt.close()

    mne.viz.plot_compare_evokeds(evokeds, picks=['Cz'], ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[1.25])
    plt.savefig(f"./figures/all_subjects_driving_MRCP_Cz.pdf")
    plt.close()
