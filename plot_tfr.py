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
            plt.savefig(f"./figures/{subject_id}_ERDS_{event}_no_baseline.pdf")
        else:
            plt.savefig(f"./figures/{subject_id}_ERDS_{event}_baseline.pdf")
        plt.close()


def main():
    raw = load_from_file(glob(f'./data/sub-P{subject_id}/motor-imagery-csp-{subject_id}-acquisition*.gdf')[0])

    raw.set_eeg_reference()
    adjust_info(raw)
    raw.set_eeg_reference()
    raw.filter(l_freq=1, h_freq=35)

    # exclude = None
    # exclude = [0, 6, 4, 10, 14, 16]
    # manual_clean_ica(raw, exclude)

    raw = mne.preprocessing.compute_current_source_density(raw)

    matplotlib.use('Agg')
    tmin = -3.0
    tmax = 5.0
    buffer = 0.5
    epochs = make_epochs(raw, tmin=tmin-buffer, tmax=tmax+buffer, include_rest=False)
    epochs.pick(["C3", "C4"])
    event_ids = dict(left=0, right=1)
    plot_tfr(epochs, baseline=(-3, 0), tmin=tmin, tmax=tmax, event_ids=event_ids)
    plot_tfr(epochs, baseline=None, tmin=tmin, tmax=tmax, event_ids=event_ids)


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    subject_ids = ["066", "587", "812", "840", "854", "942", "986"]
    for subject_id in subject_ids:
        main()
