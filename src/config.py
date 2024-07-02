import os
from pathlib import Path

ROOT_DIR = str(Path(os.path.dirname(os.path.abspath(__file__))).parents[0])
MODEL_PATH = os.path.join(ROOT_DIR, 'models') # Model Directory
MODEL_INTERIM_PATH = os.path.join(MODEL_PATH, 'interim') # Interim Model Directory
MODEL_FINAL_PATH = os.path.join(MODEL_PATH, 'final') # Final Model Directory
# DATA_PATH = os.path.join(ROOT_DIR, 'data') # Data Directory
DATA_PATH = "G:/"
METADATA_FILE = os.path.join(DATA_PATH, "all_data_updated.csv") # Metadata File
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw') # Raw Data Directory
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed') # Processed Data Directory
NORMALIZED_MEL_SPEC_PATH = os.path.join(PROCESSED_DATA_PATH, 'normalized')
PROCESSED_METADATA_FILE = os.path.join(PROCESSED_DATA_PATH, 'bee_hive_metadata.npy') # Processed Metadata File
PROCESSED_MEL_SPEC_FILE = os.path.join(PROCESSED_DATA_PATH, 'bee_hive_mel_specs.npy') # Processed Mel Spec File
TARGET_FEATURE = "queen status"

duration_of_audio_file = 60
sampling_rate = 22050
number_of_samples_in_audio_file = sampling_rate * duration_of_audio_file

n_fft = 2048
hop_length = 512
n_mels = 128
fmax = int(sampling_rate/2)

number_of_target_columns = 1