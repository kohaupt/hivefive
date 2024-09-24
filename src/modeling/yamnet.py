import torch.nn as nn
import torch
import tensorflow_hub as hub

class YamnetTransferModel(nn.Module):

  def __init__(self):
    super().__init__()

    self.model_handle = 'https://kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1'
    self.yamnet_model = hub.load(self.model_handle)
    self.num_classes = 1

    # Add layers to map the embeddings (shape [N, 1024]) to the binary classification task
    self.transfer_model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, self.num_classes)
    )

  def forward(self, x):
    # Yamnet does not support batches of audio clips, so we need to process each clip individually
    # (https://github.com/tensorflow/models/issues/10529)

    batch_results = []
    for audio_clip in x:
      # Feature extraction using YAMNet embeddings
      scores, embeddings, log_mel_spectrogram = self.yamnet_model(audio_clip)

      # Convert the embeddings to PyTorch tensor
      embeddings = embeddings.numpy()
      embeddings = torch.from_numpy(embeddings).float()

      # Pass the embeddings to the transfer learning model
      audio_clip_result = self.transfer_model(embeddings)

      # Reduce the results of each audio clip to a single value instead of N values for N frames
      audio_clip_result = torch.mean(audio_clip_result, dim=0)

      # Use sigmoid activation function to get the probability of the positive class
      audio_clip_result = torch.sigmoid(audio_clip_result)

      batch_results.append(audio_clip_result)

    batch_results = torch.stack(batch_results)

    return batch_results