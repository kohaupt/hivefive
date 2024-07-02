import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import numpy as np

class HiveDataset(Dataset):
    """
    A custom PyTorch Dataset class for chest X-Ray images.

    This class is designed to handle X-Ray image datasets, supporting both grayscale
    and RGB image modes. It allows for on-the-fly transformations of the images and labels,
    facilitating data augmentation and preprocessing steps.

    Parameters:
    - metadata (pd.DataFrame): DataFrame containing image metadata (e.g., filenames, labels).
    - img_dir (str): Directory path where images are stored.
    - classes (list): List of column names in `metadata` representing the label(s) for each image.
    - img_mode (str, optional): The mode of the images, either "RGB" or "GRAY". Default is "RGB".
    - transform (callable, optional): A function/transform that takes in an image and returns a transformed version. E.g., data augmentation procedures.
    - target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    - filename_prefix (str, optional): Prefix to add to filenames from `metadata` before loading images. Useful if `metadata` filenames do not include a common path prefix that is present in `img_dir`.

    Usage:
    dataset = XRayDataset(metadata=df, img_dir="/path/to/images", classes=['Normal', 'Pneumonia'], img_mode='RGB')
    """

    def __init__(
        self,
        # metadata_path: str,
        metadata_df: pd.DataFrame,
        # mel_spec_path: str,
        processed_data_path: str,
        target_feature: str,
        transform=None,
        fake_rgb=False
    ):
        #  metadata_column_names = ['device', 'hive number', 'date', 'hive temp', 'hive humidity',
        # 'hive pressure', 'weather temp', 'weather humidity', 'weather pressure',
        # 'wind speed', 'gust speed', 'weatherID', 'cloud coverage', 'rain',
        # 'lat', 'long', 'file name', 'queen presence', 'queen acceptance',
        # 'frames', 'target', 'time', 'queen status']
        #  metadata = np.load(metadata_path, allow_pickle=True)
        #  metadata_df = pd.DataFrame(metadata, columns=metadata_column_names)

        self.metadata = metadata_df.reset_index()
        self.target = metadata_df[target_feature].values
        self.processed_data_path = processed_data_path
        self.transform = transform
        self.fake_rgb = fake_rgb

        # self.mel_specs = np.load(mel_spec_path)

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
        mel_spec_path = os.path.join(self.processed_data_path, self.metadata.iloc[0]["file name"].replace(".wav", ".npy"))
        mel_spec = np.load(mel_spec_path)

        mel_spec = torch.tensor(mel_spec).float()
        # Bring into shape (1, 128, 5168) for CNN
        # (channels, height, width)
        mel_spec = mel_spec.unsqueeze(0)

        if(self.fake_rgb):
            # Fake 3 channels:
            mel_spec = torch.stack([mel_spec[0], mel_spec[0], mel_spec[0]], 0)

        # TODO: Do transformations here (e.g., data augmentation, normalization, etc.)
        if(self.transform):
            mel_spec = self.transform(mel_spec)
        
        # One hot encode the target
        # TODO: Is this necessary?
        # TODO: Move to preprocessing step?
        label = torch.tensor(self.target[idx]).float()
        encoded_label = torch.zeros(4)
        encoded_label[int(label)] = 1

        return mel_spec, encoded_label
