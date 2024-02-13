from glob import glob

import matplotlib
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mne.preprocessing import ICA
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test

from continuous_control_bci.data.load_data import load_from_file, adjust_info
from continuous_control_bci.data.preprocessing import make_epochs, manual_clean_ica
from continuous_control_bci.util import SUBJECT_IDS
from plot_csp_continuous import get_streams_from_xdf, CHANNEL_TYPE_MAPPING, make_rough_emg_events, subject_ids


def plot_tfr(epochs, baseline=(-2.0, -1.0), tmin=-2.0, tmax=3.75, event_ids=None):
    if event_ids is None:
        event_ids = dict(left=-1, rest=0, right=1)
    freqs = np.arange(2, 35)  # frequencies from 2-35Hz
    vmin, vmax = -1, 2  # set min and max ERDS values in plot
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

    kwargs = dict(
        n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
    )  # for cluster test

    tfr = tfr_multitaper(
        epochs,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=2,
    )
    tfr.crop(tmin, tmax)
    if baseline is not None:
        tfr.apply_baseline(baseline, mode="percent")

    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(
            1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 1]}
        )
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            if baseline is not None:
                # positive clusters
                _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
                # negative clusters
                _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

                # note that we keep clusters with p <= 0.05 from the combined clusters
                # of two independent tests; in this example, we do not correct for
                # these two comparisons
                c = np.stack(c1 + c2, axis=2)  # combined clusters
                p = np.concatenate((p1, p2))  # combined p-values
                mask = c[..., p <= 0.01].any(axis=-1)

                # plot TFR (ERDS map with masking)
                tfr_ev.average().plot(
                    [ch],
                    cmap="RdBu_r",
                    cnorm=cnorm,
                    axes=ax,
                    colorbar=False,
                    show=False,
                    mask=mask,
                    mask_style="mask",
                )
            else:
                tfr_ev.average().plot(
                    [ch],
                    cmap="RdBu_r",
                    axes=ax,
                    colorbar=False,
                )

            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({event})")
        if baseline is None:
            plt.savefig(f"./figures/{subject_id}_ERDS_driving_{event}_no_baseline.pdf")
        else:
            plt.savefig(f"./figures/{subject_id}_ERDS_driving_{event}_baseline.pdf")
        plt.close()


def main():
    fname = f"./data/sub-P{subject_id}/ses-S001/eeg/sub-P{subject_id}_ses-S001_task-Default_run-001_eeg.xdf"
    raw, eeg_stream, emg_stream, _ = get_streams_from_xdf(fname)
    raw.set_channel_types(CHANNEL_TYPE_MAPPING)
    raw.set_montage("biosemi32", on_missing='raise')
    raw.set_eeg_reference()

    raw.filter(l_freq=1, h_freq=35)

    raw = mne.preprocessing.compute_current_source_density(raw)

    matplotlib.use('Agg')
    events = make_rough_emg_events(eeg_stream, emg_stream)

    tmin = -3
    tmax = 5
    buffer = 0.5
    time_offset = 1.25

    event_ids = dict(left=-1, right=1)
    epochs = mne.Epochs(
        raw,
        events,
        event_ids,
        tmin=tmin - buffer - time_offset,
        tmax=tmax + buffer - time_offset,
        baseline=None,
        preload=True,
    )

    epochs.pick(["C3", "C4"])
    epochs.shift_time(time_offset)

    event_ids = dict(left=0, right=1)
    plot_tfr(epochs, baseline=(-3, 0), tmin=tmin, tmax=tmax, event_ids=event_ids)
    plot_tfr(epochs, baseline=None, tmin=tmin, tmax=tmax, event_ids=event_ids)


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    for subject_id in SUBJECT_IDS:
        main()
