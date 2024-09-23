import os
from pathlib import Path

ROOT_DIR = str(Path(os.path.dirname(os.path.abspath(__file__))).parents[0])
MODEL_PATH = os.path.join(ROOT_DIR, 'models') # Model Directory
MODEL_INTERIM_PATH = os.path.join(MODEL_PATH, 'interim') # Interim Model Directory
MODEL_FINAL_PATH = os.path.join(MODEL_PATH, 'final') # Final Model Directory

# DATA_PATH = os.path.join(ROOT_DIR, 'data') # Data Directory
DATA_PATH = "G:/"
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw') # Raw Data Directory
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed') # Processed Data Directory
NORMALIZED_MEL_SPEC_PATH = os.path.join(PROCESSED_DATA_PATH, 'normalized')
METADATA_FILE = os.path.join(RAW_DATA_PATH, 'state_labels.csv') # Metadata File
PROCESSED_METADATA_FILE = os.path.join(PROCESSED_DATA_PATH, 'bee_hive_metadata.npy') # Processed Metadata File
PROCESSED_METADATA_FILE_SEGMENTED = os.path.join(PROCESSED_DATA_PATH, 'bee_hive_metadata_segmented.npy') # Processed Metadata File
TARGET_FEATURE = "label"

duration_of_audio_file = 60
sampling_rate = 22050
number_of_samples_in_audio_file = sampling_rate * duration_of_audio_file

# Segmentation configuration
segment_duration = 60.0
overlap_duration = 1.0

# Mel spectrogram configuration
n_fft = 2048
hop_length = 512
n_mels = 128
fmax = 5000

number_of_target_columns = 1