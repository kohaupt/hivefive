import os
import numpy as np
import librosa
import pandas as pd
import config


def extract_target_from_metadata():
    data = pd.read_csv(os.path.join(
        config.path_data, config.metadata_filename))

    return data[["queen status", "file name"]]


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


def preprocess_data_and_pack_to_npy(audio_files_path=config.RAW_DATA_PATH, processed_data_path=config.PROCESSED_DATA_PATH, metadata_file_path=config.METADATA_FILE):
    """Preprocesses the audio files and metadata and packs them into a .npy file.

    Parameters:
    - audio_files_path (str): Path to the directory containing the audio files.
    - processed_data_path (str): Path to the directory where the .npy file should be saved.
    """

    print('Packing audio files to .npy file...')
    print('Audio files path:', audio_files_path)
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    packed_np_path_mel_specs = os.path.join(processed_data_path, 'bee_hive_mel_specs.npy')
    packed_np_path_metadata = os.path.join(processed_data_path, 'bee_hive_metadata.npy')

    audio_files = librosa.util.find_files(audio_files_path, ext=['wav'])
    len_audio_files = len(audio_files)
    print('Number of audio files:', len_audio_files)

    metadata_df = pd.read_csv(metadata_file_path)

    mel_specs = []
    metadatas = []
    for index, file in enumerate(audio_files):
        print('--- Preparing file number:', index + 1, 'of', len_audio_files, '---')

        # Get corresponding metadata for the audio file
        metadata_key = get_matching_raw_filename(file)
        metadata = metadata_df.loc[metadata_df['file name'] == metadata_key]
        metadata = np.array(metadata)
        metadata = np.squeeze(metadata)
        metadata[16] = os.path.basename(file) # replace the raw file name with the "real" "file name

        # Generate mel spec
        audio_data, _ = librosa.load(
            file, duration=config.duration_of_audio_file, sr=None)
        # audio_data = librosa.resample(y=audio_data, orig_sr=22050, target_sr=config.sampling_rate)
        mel_spectrogram = create_mel_spectrogram_from_audio_data(audio_data, hop_length=256)

        mel_spec_array = np.array(mel_spectrogram)
        path_mel_spec_file = os.path.join(processed_data_path, os.path.basename(file).replace('.wav', ''))
        np.save(path_mel_spec_file, mel_spec_array)

        # mel_specs.append(mel_spectrogram)
        metadatas.append(metadata)

    # Write the data to two .npy files
    # mel_specs = np.array(mel_specs, dtype=np.float16)
    metadata = np.array(metadatas)
    
    # np.save(packed_np_path_mel_specs, mel_specs)
    np.save(packed_np_path_metadata, metadata)


if __name__ == "__main__":
    preprocess_data_and_pack_to_npy()