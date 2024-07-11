import os
import numpy as np
import librosa
import pandas as pd
import config
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import re
import glob


def get_matching_raw_filename(file):
    filename = os.path.basename(file)
    filename_no_extension = os.path.splitext(filename)[0]
    filename_no_segment = filename_no_extension.split('__')[0]
    matching_raw_filename = filename_no_segment + '.raw'
    return matching_raw_filename


def segment_audio(file_path, segment_duration=config.segment_duration, overlap_duration=config.overlap_duration):
    """Segments the audio file at the given path into smaller segments with a fixed duration and overlap. 
    Also normalizes the audio data. Padding is added if the last segment is shorter than the fixed duration."""
    audio, sr = librosa.load(file_path)

    # Normalize the mel spectrogram
    # TODO: We could also use audio = librosa.util.normalize(audio) instead of MinMaxScaler here (?)
    # scaler = MinMaxScaler()
    # scaler.fit(audio)
    # mel_spectrogram = scaler.transform(audio)
    audio = librosa.util.normalize(audio)
    print("Mel spec min:", np.min(audio), "Mel spec max:", np.max(audio))

    segment_length = int(segment_duration * sr)
    overlap_length = int(overlap_duration * sr)

    segments = []
    for start in range(0, len(audio), segment_length):
        end = start + segment_length + overlap_length  # Add overlap
        segment = audio[start:end]

        if len(segment) < segment_length:
            # pad with zeros = silence
            segment = np.pad(
                segment, (0, (segment_length + overlap_length) - len(segment)))

        segments.append(segment)
    return segments


def create_mel_spectrogram_from_audio_data(audio_data: np.ndarray, sampling_rate=config.sampling_rate, hop_length=config.hop_length, n_mels=config.n_mels):
    """Creates a mel spectrogram from the given audio data."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data, sr=sampling_rate, hop_length=hop_length, n_mels=n_mels, fmax=int(sampling_rate/2), power=2)

    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram

def add_metadata_to_segment(metadata_df: pd.DataFrame, filename: str):
    """Adds an entry to the metadata DataFrame for the given segment(name)

    Parameters:
    - metadata_df (pd.DataFrame): The dataframe containing the metadata.
    - metadata_index (int): The filename of the segment to add the metadata for.

    Returns:
    - metadata_df (pd.DataFrame): The updated metadata dataframe.
    """

    segment_number = filename.split("__segment")[1]
    segment_mask = metadata_df["sample_name"].str.contains(filename.split("__segment")[0])

    # If it is the first segment, we just add the sample index to the sample name
    if int(segment_number) == 0:
        datapoint_index = metadata_df[segment_mask].index[0]
        metadata_df.at[datapoint_index, "sample_name"] = filename
        return metadata_df

    # If it is not the first segment, we copy the entry and add the sample index to the sample name
    new_row = metadata_df[segment_mask].iloc[0].copy()
    new_row["sample_name"] = filename
    metadata_df.loc[len(metadata_df)] = new_row

    print("Added metadata entry for segment " + filename)
    return metadata_df


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
        try:

            file_basename = os.path.basename(file).replace('.wav', '')
            path_mel_spec_file = os.path.join(processed_data_path, file_basename)

            if os.path.isfile(path_mel_spec_file + "__segment0" + '.npy'):
                print("File seems to be existing & processed. Skipping " + file_basename)
                continue

            print('--- Preparing file number:', index + 1, 'of', len_audio_files, '---')

            # Generate mel spec
            segments = segment_audio(file, config.segment_duration, config.overlap_duration)

            for idx, segment in enumerate(segments):
                mel_spectrogram = create_mel_spectrogram_from_audio_data(segment, hop_length=256)

                mel_spec_array = np.array(mel_spectrogram)
                np.save(path_mel_spec_file + "__segment" + str(idx), mel_spec_array)

        except BaseException:
            print("Error: Skipping file " + file + " due to a runtime error.")
            continue

    print('---------------------------------------------')
    print('Finished packing audio files to .npy files.')
    print('---------------------------------------------')


def preprocess_metadata():
    """Preprocesses the metadata file and saves it as a .npy file.
    
    Changes made to the metadata file:
    - Extracts the hive number from the sample name and saves it as a separate column.
    - Encodes the target feature via label encoding.
    """

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


def segment_metadata():
    """Segments the metadata file and saves it as a .npy file.
    """

    print('Adding metadata entries for segments...')
    metadata_column_names = ['sample_name', "label", "hive number"]
    metadata = np.load(config.PROCESSED_METADATA_FILE, allow_pickle=True)
    metadata_df = pd.DataFrame(metadata, columns=metadata_column_names)

    files = glob.glob(os.path.join(config.NORMALIZED_MEL_SPEC_PATH, "*.npy"))
    print("Found: ", len(files), " files. Processing...")

    segment_count = 0
    for _, file in enumerate(files):
        filename = os.path.basename(file).replace('.npy', '')
        file_basename = filename.split("__segment")[0]

        if metadata_df[metadata_df['sample_name'].str.contains(file_basename)].empty:
            print("Could not find metadata entry for file " + filename)
            print("Skipping file " + filename)
            continue

        metadata_df = add_metadata_to_segment(metadata_df, filename)
        segment_count += 1

    np.save(config.PROCESSED_METADATA_FILE_SEGMENTED, metadata_df.to_numpy())
    print('---------------------------------------------')
    print('Finished adding', segment_count, 'metadata entries for', len(files), 'files.')
    print('---------------------------------------------')


if __name__ == "__main__":
    preprocess_metadata()
    preprocess_data_and_pack_to_npy()
    segment_metadata()