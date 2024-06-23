import os

ROOT_DIR = os.path.dirname("D:\\Software-Projekte\\Uni\\ds_audio\\") # Project Root Directory
DATA_PATH = os.path.join(ROOT_DIR, 'data') # Data Directory
METADATA_PATH = os.path.join(DATA_PATH, "all_data_updated.csv") # Metadata File
DATA_SRC_PATH = os.path.join(DATA_PATH, 'raw') # Raw Data Directory
TARGET_DIR = os.path.join(DATA_PATH, 'processed') # Processed Data Directory
TARGET_FEATURE = "queen status"

duration_of_audio_file = 60
sampling_rate = 22050
number_of_samples_in_audio_file = sampling_rate * duration_of_audio_file

n_fft = 2048
hop_length = 256
n_mels = 128
fmax = int(sampling_rate/2)

number_of_target_columns = 1