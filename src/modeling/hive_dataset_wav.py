import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import soundfile as sf
import resampy

import sys
from pathlib import Path
# Allow imports from the src directory
sys.path.append(
    str(Path(os.path.dirname(os.path.abspath(__file__))).parents[0]))

class HiveDatasetWav(Dataset):
    """
    A custom PyTorch Dataset class for bee hive audio data.

    This class loads the audio data from the segmented WAV files and their corresponding labels.

    Attributes:
    - metadata_df (pd.DataFrame): A DataFrame containing the metadata of the audio data.
    - segmented_data_path (str): The path to the directory containing the segmented WAV files.
    - target_feature (str): The name of the target feature in the metadata DataFrame.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        segmented_data_path: str,
        target_feature: str,
    ):
        self.metadata = metadata_df.reset_index()
        self.target = metadata_df[target_feature].values
        self.segmented_data_path = segmented_data_path

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """Retrieves the mel_spec and its label at the specified index `idx`.

        Parameters:
        - idx (int): Index of the item to retrieve.

        Returns:
        - tuple: (mel_spec, label) where image is the mel_spec and label is the target(s).
        """
        # Load wav file and resample it to 16000 Hz
        # Using the preprocessing steps from the original source code
        # (https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/inference.py)

        # Decode the WAV file.
        wav_path = os.path.join(self.segmented_data_path, self.metadata.iloc[idx]["segment"] + ".wav")
        wav_data, sr = sf.read(wav_path, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
          waveform = np.mean(waveform, axis=1)
        if sr != 16000:
          waveform = resampy.resample(waveform, sr, 16000)

        label = torch.tensor(self.target[idx]).float()

        return waveform, label