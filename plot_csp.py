import matplotlib.pyplot as plt
import mne
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from continuous_control_bci.data.load_data import load_from_file, adjust_info
from continuous_control_bci.data.preprocessing import apply_causal_filters, make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier


def main():
    raw = load_from_file("./data/pilot_1/calibration/horse_reighns_pilot_driving.gdf")
    raw = raw.set_eeg_reference()
    raw = adjust_info(raw)
    raw = apply_causal_filters(raw)
    epochs = make_epochs(raw)
    X_train, _, y_train, _ = epochs_to_train_test(epochs)
    print("Training classifier. This may take a while..")
    clf, y_pred = create_csp_classifier(X_train, y_train)
    print(classification_report(y_train, y_pred, target_names=["Left", "Right", "Rest"]))
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, display_labels=["Left", "Right", "Rest"],
                                            normalize='true')
    plt.title("Confusion matrix on calibration data")
    plt.savefig("./figures/confusion_matrix_calibration.pdf")
    plt.show()

    csp = clf.steps[0][1]
    csp.plot_filters(info=raw.pick('eeg').info, show_names=True, sphere='eeglab', colorbar=False)
    plt.savefig("./figures/csp_filters.pdf")
    plt.show()


if __name__ == "__main__":
    mne.set_log_level('warning') # noqa
    main()
