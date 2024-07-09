import os
import numpy as np
import librosa
import pandas as pd
import config
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import re


def get_matching_raw_filename(file):
    filename = os.path.basename(file)
    filename_no_extension = os.path.splitext(filename)[0]
    filename_no_segment = filename_no_extension.split('__')[0]
    matching_raw_filename = filename_no_segment + '.raw'
    return matching_raw_filename


def create_mel_spectrogram_from_audio_data(audio_data: np.ndarray, sampling_rate=config.sampling_rate, hop_length=config.hop_length, n_mels=config.n_mels):
    """Creates a mel spectrogram from the given audio data."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data, sr=sampling_rate, hop_length=hop_length, n_mels=n_mels, fmax=int(sampling_rate/2), power=2)

    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram


def preprocess_data_and_pack_to_npy(audio_files_path=config.RAW_DATA_PATH, processed_data_path=config.NORMALIZED_MEL_SPEC_PATH):
    """Preprocesses the audio files packs each of them into a .npy file.

    Parameters:
    - audio_files_path (str): Path to the directory containing the audio files.
    - processed_data_path (str): Path to the directory where the .npy file should be saved.
    """

    print('Packing audio files to .npy files...')
    print('Audio files path:', audio_files_path)
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    audio_files = librosa.util.find_files(audio_files_path, ext=['wav'])
    len_audio_files = len(audio_files)
    print('Number of audio files:', len_audio_files)

    for index, file in enumerate(audio_files):
        print('--- Preparing file number:', index + 1, 'of', len_audio_files, '---')

        # Generate mel spec
        audio_data, _ = librosa.load(
            file, duration=config.duration_of_audio_file, sr=None)
        mel_spectrogram = create_mel_spectrogram_from_audio_data(audio_data, hop_length=256)

        # Normalize the mel spectrogram
        # TODO: We could also use librosa.util.normalize() instead of MinMaxScaler here (?)
        scaler = MinMaxScaler()
        scaler.fit(mel_spectrogram)
        mel_spectrogram = scaler.transform(mel_spectrogram)
        print("Mel spec min:", np.min(mel_spectrogram), "Mel spec max:", np.max(mel_spectrogram))

        mel_spec_array = np.array(mel_spectrogram)
        path_mel_spec_file = os.path.join(processed_data_path, os.path.basename(file).replace('.wav', ''))
        np.save(path_mel_spec_file, mel_spec_array)


def preprocess_metadata():
    metadata_df = pd.read_csv(config.METADATA_FILE)

    # Extract hive number from sample name and save it as a separate column
    metadata_df["hive number"] = 0

    for index, row in metadata_df.iterrows():
        hive_number = re.match("Hive\d", row["sample_name"]).group(0)[-1:]
        metadata_df.at[index, "hive number"] = int(hive_number)

    # # TODO: Remove this once we use the full dataset
    # # Select a subset of the data with only one beehive (get hive number from filename)
    # metadata_df = metadata_df[metadata_df["hive number"] == "Hive1"]

    # Encode the target feature via label encoding
    le = LabelEncoder()
    metadata_df["label"] = le.fit_transform(metadata_df["label"])

    np.save(config.PROCESSED_METADATA_FILE, metadata_df.to_numpy())


if __name__ == "__main__":
    preprocess_metadata()
    preprocess_data_and_pack_to_npy()