from dataclasses import dataclass
from typing import Optional

import mne.io


@dataclass
class DrivingStreams:
    raw: mne.io.RawArray
    eeg_stream: dict
    emg_prediction_stream: dict
    unity_stream: dict
    position_stream: Optional[dict]
