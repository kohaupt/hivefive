import os
import numpy as np
import h5py
import librosa
import config
import pandas as pd


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
