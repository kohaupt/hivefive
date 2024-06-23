import features
import config

# Pack audio files to HDF5 file
audio_files_path = './data/raw'
hdf5_file_path = './data/processed'
# features.compute_melspectrogram(src_path=audio_files_path, target_path=hdf5_file_path)

features.preprocess_data_and_pack_to_npy(config.DATA_SRC_PATH, config.TARGET_DIR, config.METADATA_PATH)
