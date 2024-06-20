import os
import numpy as np
import h5py
import librosa
import config
import pandas as pd
import gc
import matplotlib.pyplot as plt


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
        y=audio_data, sr=sampling_rate, hop_length=hop_length, n_mels=n_mels, fmax=int(sampling_rate/2))

    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram


def preprocess_data_and_pack_to_npy(audio_files_path, processed_data_path, metadata_path):
    """Preprocesses the audio files and metadata and packs them into a .npy file.

    Parameters:
    - audio_files_path (str): Path to the directory containing the audio files.
    - processed_data_path (str): Path to the directory where the .npy file should be saved.
    """

    print('Packing audio files to .npy file...')

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    packed_np_path_mel_specs = os.path.join(processed_data_path, 'bee_hive_mel_specs.npy')
    packed_np_path_metadata = os.path.join(processed_data_path, 'bee_hive_metadata.npy')


    audio_files = librosa.util.find_files(audio_files_path, ext=['wav'])
    len_audio_files = len(audio_files)
    print('Number of audio files:', len_audio_files)

    metadata_df = pd.read_csv(metadata_path)

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
        mel_spectrogram = create_mel_spectrogram_from_audio_data(audio_data, hop_length=256)

        mel_specs.append(mel_spectrogram)
        metadatas.append(metadata)

    # Write the data to two .npy files
    mel_specs = np.array(mel_specs)
    metadata = np.array(metadatas)
    
    np.save(packed_np_path_mel_specs, mel_specs)
    np.save(packed_np_path_metadata, metadata)


# preprocess_data_and_pack_to_npy("./data/raw", "./data/processed", "./data/all_data_updated.csv")


def pack_audio_files_to_hdf5(audio_files_path, hdf5_file_path):
    if not os.path.exists(hdf5_file_path):
        os.makedirs(hdf5_file_path)
    packed_hdf5_path = os.path.join(hdf5_file_path, 'bee_hive.h5')

    audio_files = librosa.util.find_files(audio_files_path, ext=['wav'])
    len_audio_files = len(audio_files)

    target_and_filename = extract_target_from_metadata()

    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name',
            shape=(len_audio_files,),
            dtype='S80')  # S80 is a string of 80 characters (max length of a file name)

        hf.create_dataset(
            name='audio_data',
            shape=(len_audio_files, config.number_of_samples_in_audio_file),
            dtype=np.float32)

        hf.create_dataset(
            name='target',
            shape=(len_audio_files, config.number_of_target_columns),
            dtype=np.float32)

        for index, file in enumerate(audio_files):
            print('--- Packing file number:', index,
                  'of', len_audio_files, '---')
            filename = get_matching_raw_filename(file)
            # Load audio file and make sure it has the correct duration and sampling rate
            audio_data, sample_rate = librosa.load(
                file, duration=config.duration_of_audio_file, sr=config.sampling_rate)
            target_for_file = target_and_filename.loc[target_and_filename['file name']
                                                      == filename][config.target_feature].values[0]

            hf['audio_name'][index] = filename
            hf['audio_data'][index] = audio_data
            hf['target'][index] = target_for_file


def compute_melspectrogram(audio_data):
    # fig, ax = plt.subplots(1, 1, figsize=(7.50, 3.50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 3150 Hz (int(sample_rate/7)) is set as the max frequency for the spectrogram
    ms = librosa.feature.melspectrogram(y=audio_data, sr=config.sampling_rate, n_fft=2048, hop_length=256, n_mels=128, fmax=int(config.sampling_rate/7))
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.util.normalize(log_ms)
    return log_ms
    # librosa.display.specshow(log_ms, sr=sample_rate)

    # fig.savefig(image_filename)
    # plt.close(fig)
    # del fig, ax, ms, log_ms
    # gc.collect()


# src_path: path to the directory containing the audio files
# target_path: path to the directory where the spectrograms will be saved
# batch_size: number of audio files to process at once
# limit: number of audio files to process, None if all files should be processed
def compute_melspectrogram(src_path, target_path, batch_size=50, offset=0, limit=None):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    packed_hdf5_path = os.path.join(target_path, 'bee_hive.h5')

    target_and_filename = extract_target_from_metadata()
    audio_files = librosa.util.find_files(src_path, ext=['wav'])
    len_audio_files = len(audio_files)

    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name',
            shape=(len_audio_files,),
            dtype='S80')  # S80 is a string of 80 characters (max length of a file name)

        hf.create_dataset(
            name='audio_data',
            shape=(len_audio_files, config.number_of_samples_in_audio_file),
            dtype=np.float32)

        hf.create_dataset(
            name='target',
            shape=(len_audio_files, config.number_of_target_columns),
            dtype=np.float32)

        if offset > 0:
            print(f"Skipping {offset} files.")

        for i in range(offset, len(audio_files), batch_size):
            for j in range(i, min(i+batch_size, len(audio_files))):
                if j == limit:
                    return

                # Retrieve the filename without the path
                basename = os.path.basename(audio_files[j])
                # Remove the extension
                target_filename = os.path.splitext(basename)[0]
                filename_raw = get_matching_raw_filename(audio_files[j])

                if hf['audio_name'].contains(target_filename):
                    print(f"Skip file {target_filename} (file {j}/{len(audio_files)}): Already exists.")
                    continue

                audio_data = librosa.load(audio_files[j], duration=config.duration_of_audio_file, sr=config.sampling_rate)
                melspectrogram = compute_melspectrogram(audio_files[j])

                target_for_file = target_and_filename.loc[target_and_filename['file name']
                                                      == filename_raw][config.target_feature].values[0]
                
                hf['audio_name'][j] = target_filename
                hf['audio_data'][j] = melspectrogram
                hf['target'][j] = target_for_file
                print(f"Created spectogram for {target_filename} (file {j}/{len(audio_files)}).")

            print(f"--- Checkpoint: Processed {min(i+batch_size, len(audio_files))} files. ---")
            del audio_time_series
            gc.collect()
            
        print("--- SUCCESS: All files processed. ---")
