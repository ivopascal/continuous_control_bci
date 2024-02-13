from glob import glob

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from continuous_control_bci.data.load_data import load_from_file, adjust_info
from continuous_control_bci.data.preprocessing import apply_causal_filters, make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier
from sklearn.metrics import f1_score

from continuous_control_bci.util import SUBJECT_IDS


def main():
    # raw = load_from_file("./data/pilot_1/calibration/horse_reighns_pilot_driving.gdf")
    raw = load_from_file(glob(f'./data/sub-P{subject_id}/motor-imagery-csp-{subject_id}-acquisition*.gdf')[-1])

    raw = adjust_info(raw)
    raw = raw.set_eeg_reference(ref_channels=["Cz"], ch_type='eeg')
    raw.drop_channels('Cz')

    # raw = apply_causal_filters(raw)
    raw.filter(l_freq=5, h_freq=35)

    include_rest = True
    epochs = make_epochs(raw, include_rest=include_rest)
    X_train, _, y_train, _ = epochs_to_train_test(epochs)
    print("Training classifier. This may take a while..")
    clf, y_pred = create_csp_classifier(X_train, y_train)
    print("Classifier trained!")
    if include_rest:
        target_names = ["Left", "Right", "Rest"]
    else:
        target_names = ["Left", "Right"]

    print(f"Subject {subject_id}")
    print(classification_report(y_train, y_pred, target_names=target_names))
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, display_labels=target_names,
                                            normalize='true')
    f1 = f1_score(y_train, y_pred, average='macro')
    plt.title("Confusion matrix on calibration data")
    plt.savefig(f"./figures/{subject_id}_confusion_matrix_calibration.pdf")
    plt.close()

    csp = clf.steps[0][1]
    csp.plot_filters(info=raw.pick('eeg').info, show_names=True, sphere='eeglab', colorbar=False)
    plt.savefig(f"./figures/{subject_id}_csp_filters.pdf")
    plt.close()

    return f1


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa

    f1s = []
    for subject_id in SUBJECT_IDS:
        f1s.append(main())

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)
