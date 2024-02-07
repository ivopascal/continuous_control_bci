from glob import glob

import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from continuous_control_bci.data.emg_events import make_rough_emg_events
from continuous_control_bci.data.load_data import load_from_file, adjust_info, get_driving_epochs_for_csp
from continuous_control_bci.data.preprocessing import apply_causal_filters, make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier
from sklearn.metrics import f1_score

from continuous_control_bci.util import emg_classes_to_eeg_classes
from plot_csp_continuous import get_streams_from_xdf, CHANNEL_TYPE_MAPPING





def main():
    raw = load_from_file(glob(f'./data/sub-P{subject_id}/motor-imagery-csp-{subject_id}-acquisition*.gdf')[0])

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

    driving_epochs = get_driving_epochs_for_csp(subject_id, include_rest)
    X_driving = driving_epochs.get_data(copy=True, picks=['eeg'])
    y_driving = emg_classes_to_eeg_classes(driving_epochs.events[:, -1])

    y_driving_pred = clf.predict(X_driving)
    print(f"Transfer performance")
    print(classification_report(y_driving, y_driving_pred, target_names=target_names))

    ConfusionMatrixDisplay.from_predictions(y_driving, y_driving_pred, display_labels=target_names,
                                            normalize='true')
    f1 = f1_score(y_driving, y_driving_pred, average='macro')
    plt.title("Confusion matrix on transfer")
    plt.savefig(f"./figures/{subject_id}_confusion_matrix_transfer.pdf")
    plt.close()

    return f1


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa

    subject_ids = ["066", "587", "812", "840", "854", "942", "986"]

    f1s = []
    for subject_id in subject_ids:
        f1s.append(main())

    print(np.mean(f1s))
    print(np.std(f1s))
    print(f1s)
