import features as features

# Pack audio files to HDF5 file
audio_files_path = './data/raw'
hdf5_file_path = './data/processed'
features.pack_audio_files_to_hdf5(audio_files_path, hdf5_file_path)
