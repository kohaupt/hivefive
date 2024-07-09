import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import torch.nn.functional as F

class HiveDataset(Dataset):
    """
    A custom PyTorch Dataset class for bee hive mel spectrogram images. This class is designed to handle mel spectrogram datasets.

    Parameters:
    - metadata (pd.DataFrame): DataFrame containing image metadata (e.g., filenames, labels).
    - processed_data_path (str): Directory path where processed mel specs are stored (requires .npy files).
    - target_feature (str): Column name in `metadata` representing the label for each mel spec.
    - transform (callable, optional): A function/transform that takes in an image and returns a transformed version. E.g., data augmentation procedures.
    - fake_rgb (bool): If True, the mel spec will be converted to a 3-channel image (RGB) by stacking the same image 3 times.

    Usage:
    dataset = HiveDataset(metadata=df, processed_data_path="/path/to/images", target_feature='queen_status', fake_rgb=True)
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        processed_data_path: str,
        target_feature: str,
        transform=None,
        fake_rgb=False
    ):
        self.metadata = metadata_df.reset_index()
        self.target = metadata_df[target_feature].values
        self.processed_data_path = processed_data_path
        self.transform = transform
        self.fake_rgb = fake_rgb

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
        # Load mel_spec
        mel_spec_path = os.path.join(self.processed_data_path, self.metadata.iloc[idx]["sample_name"] + ".npy")
        mel_spec = np.load(mel_spec_path)

        mel_spec = torch.tensor(mel_spec).float()

        # Bring into shape (1, 128, 5168) for CNN
        # (channels, height, width)
        mel_spec = mel_spec.unsqueeze(0)

        # TODO: Should we remove this?
        # Scale the time dimension to 400
        mel_spec = F.interpolate(mel_spec.unsqueeze(0), size=(128, 400), mode='bilinear', align_corners=False)

        # TODO: Find a smarter way to do this
        # Remove the extra dimension added by the interpolation step (1, 1, 128, 400) -> (1, 128, 400)
        mel_spec = mel_spec.squeeze(0)

        # Fake RGB to be able to use pre-trained models
        if(self.fake_rgb):
            # Fake 3 channels:
            mel_spec = torch.stack([mel_spec[0], mel_spec[0], mel_spec[0]], 0)

        if(self.transform):
            mel_spec = self.transform(mel_spec)

        label = torch.tensor(self.target[idx]).float()

        return mel_spec, label
