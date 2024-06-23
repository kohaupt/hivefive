
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
from tqdm.notebook import tqdm
from model import CNNModel
import numpy as np
from hive_dataset import HiveDataset
from hive_dataset_npy import HiveDatasetNPY
import os

import sys
sys.path.append("D:\\Software-Projekte\\Uni\\ds_audio\\src\\")
import config

class Train():

    def __init__(self):
        print(config.fmax)
        dataset = HiveDatasetNPY(metadata_path=os.path.join(config.TARGET_DIR, "bee_hive_metadata.npy"), img_dir=os.path.join(config.TARGET_DIR, "bee_hive_mel_specs.npy"), classes=[config.TARGET_FEATURE], img_mode='RGB')
        self.training_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

        self.loss_fn = nn.BCELoss()

        input = nn.Sigmoid()(torch.randn(5, requires_grad=True))
        target = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float)
        self.loss_fn(input, target)
        self.loss_fn = nn.BCELoss()

        self.model = CNNModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def train_one_epoch(self, epoch_index, tb_writer, device="cpu"):
        running_loss = 0.
        avg_loss = 0.

        for i, data in enumerate(tqdm(self.training_dataloader)):
            
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs.to(device))

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            avg_loss += loss.item()
            if (i + 1) % self.training_dataloader.batch_size == 0:
                last_loss = running_loss / self.training_dataloader.batch_size  # loss per batch
                print(f"  batch {i +1} loss: {last_loss:.3f}")
                tb_x = epoch_index * len(self.training_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return avg_loss / (i + 1)


    # Initializing in a separate cell so we can easily add more epochs to the same run
    def train_cnn(self):
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
            self.model.train(True)
            avg_loss = self.train_one_epoch(self, epoch_number, writer, device)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.training_dataloader):
                    if i > 10:
                        break
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs.to(device))
                    vloss = self.loss_fn(voutputs, vlabels)
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
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1



trainer = Train()
trainer.train_cnn()