
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
from tqdm.notebook import tqdm
from model import CNNModel
import numpy as np
from hive_dataset import HiveDataset
import pandas as pd


model = CNNModel()

df = pd.read_csv("../../data/all_data_updated.csv")
dataset = HiveDataset(metadata=df, img_dir="../../data/processed/", classes=['queen status'], img_mode='RGB')
training_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

loss_fn = nn.BCELoss()

input = nn.Sigmoid()(torch.randn(5, requires_grad=True))
target = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float)
loss_fn(input, target)
loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train_one_epoch(model, epoch_index, tb_writer, training_dataloader, device="cpu"):
    running_loss = 0.
    avg_loss = 0.

    for i, data in enumerate(tqdm(training_dataloader)):
        
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs.to(device))

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss += loss.item()
        if (i + 1) % training_dataloader.batch_size == 0:
            last_loss = running_loss / training_dataloader.batch_size  # loss per batch
            print(f"  batch {i +1} loss: {last_loss:.3f}")
            tb_x = epoch_index * len(training_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return avg_loss / (i + 1)


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/cnn_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 2

best_vloss = np.inf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training will happen on {device}.")

for epoch in range(EPOCHS):
    print(f"EPOCH {epoch_number + 1}")

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(model, epoch_number, writer, training_dataloader, device)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(training_dataloader):
            if i > 10:
                break
            vinputs, vlabels = vdata
            voutputs = model(vinputs.to(device))
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f"LOSS train {avg_loss:.3f} valid {avg_vloss:.3f}")

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"model_{timestamp}_{epoch_number}"
        torch.save(model.state_dict(), model_path)

    epoch_number += 1