import features as features

# Pack audio files to HDF5 file
audio_files_path = './data/raw'
hdf5_file_path = './data/processed'
features.compute_melspectrogram(src_path=audio_files_path, target_path=hdf5_file_path)
