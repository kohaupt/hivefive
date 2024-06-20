path_data = "./data"
path_raw_data = "/raw"
metadata_filename = "all_data_updated.csv"
target_feature = "queen status"

duration_of_audio_file = 60
sampling_rate = 11025
number_of_samples_in_audio_file = sampling_rate * duration_of_audio_file

n_fft = 2048
hop_length = 256
n_mels = 128
fmax = int(sampling_rate/2)

number_of_target_columns = 1