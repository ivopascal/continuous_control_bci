from glob import glob

import matplotlib
import mne
from matplotlib import pyplot as plt
from tqdm import tqdm

from continuous_control_bci.data.load_data import load_from_file, adjust_info
from continuous_control_bci.data.preprocessing import make_epochs, manual_clean_ica, CALIBRATION_MRCP_BAD_ICS
from continuous_control_bci.util import SUBJECT_IDS


def main():
    raw = load_from_file(glob(f'./data/sub-P{subject_id}/motor-imagery-csp-{subject_id}-acquisition*.gdf')[0])

    raw = adjust_info(raw)
    # raw.filter(l_freq=0.05, h_freq=100)
    raw = raw.filter(0.1, 50, picks='eeg', method='iir', phase='forward')
    manual_clean_ica(raw, subject_id=subject_id, ic_dict=CALIBRATION_MRCP_BAD_ICS)

    matplotlib.use('Agg')

    # raw = raw.set_eeg_reference(['Cz'])
    raw = raw.set_eeg_reference(["Fz", 'Pz'])

    # auto_clean_ica(raw)
    raw = raw.filter(0.1, 3, picks='eeg', method='iir', phase='forward')
    # raw = raw.filter(0.1, 3)
    # raw = mne.preprocessing.compute_current_source_density(raw)
    # raw.set_channel_types(CHANNEL_TYPE_MAPPING)
    # raw.pick(["C3", "C4"])
    epochs = make_epochs(raw, tmin=-0.9, tmax=5, include_rest=False)
    # epochs.plot()
    # epochs.apply_baseline((-0.9, 0))

    # evokeds_average = epochs.average(by_event_type=True)

    # times = np.arange(-0.25, 5, 0.5)
    # evokeds_average[0].plot_topomap(times=times, average=0.5)
    # fig = plt.gcf()
    # fig.suptitle(f'Evoked left hand {subject_id}')
    # plt.show()
    #
    # evokeds_average[1].plot_topomap(times=times, average=0.5)
    # fig = plt.gcf()
    # fig.suptitle(f'Evoked right hand {subject_id}')
    # plt.show()

    # print(subject_id)
    # for evk in evokeds_average:
    #     evk.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))

    # for evoked in evokeds_average:
    #     evoked.plot(picks=['C3', 'Cz', 'C4'])
    #     plt.title(f"{evoked.comment}")

    evokeds = dict(
        left=list(epochs["left"].iter_evoked()),
        right=list(epochs["right"].iter_evoked()),
        # rest=list(epochs["rest"].iter_evoked()),
    )

    print(subject_id)
    ylim = [-10, 10]
    #
    # mne.viz.plot_compare_evokeds(evokeds, axes='topo', ylim=dict(eeg=ylim, csd=ylim),
    #                              vlines=[0.0, 1.25])
    #
    # plt.savefig(f"./figures/{subject_id}_MRCP.pdf")
    # plt.close()
    #
    # mne.viz.plot_compare_evokeds(evokeds, picks=['C3'], ylim=dict(eeg=ylim, csd=ylim),
    #                              vlines=[0.0, 1.25])
    # plt.savefig(f"./figures/{subject_id}_MRCP_C3.pdf")
    # plt.close()
    #
    # mne.viz.plot_compare_evokeds(evokeds, picks=['C4'], ylim=dict(eeg=ylim, csd=ylim),
    #                              vlines=[0.0, 1.25])
    # plt.savefig(f"./figures/{subject_id}_MRCP_C4.pdf")
    # plt.close()

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
                                 vlines=[0.0, 1.25])

    plt.savefig(f"./figures/all_subjects_MRCP.pdf")
    plt.close()

    mne.viz.plot_compare_evokeds(evokeds, picks=['C3'], ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[0.0, 1.25])
    plt.savefig(f"./figures/all_subjects_MRCP_C3.pdf")
    plt.close()

    mne.viz.plot_compare_evokeds(evokeds, picks=['C4'], ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[0.0, 1.25])
    plt.savefig(f"./figures/all_subjects_MRCP_C4.pdf")
    plt.close()

    mne.viz.plot_compare_evokeds(evokeds, picks=['Cz'], ylim=dict(eeg=ylim, csd=ylim),
                                 vlines=[0.0, 1.25])
    plt.savefig(f"./figures/all_subjects_MRCP_Cz.pdf")
    plt.close()
