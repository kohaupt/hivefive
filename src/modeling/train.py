
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchinfo import summary
from datetime import datetime
from tqdm import tqdm
from model import CNNModel
from model_transfer_learning import DenseNetTransferModel
import numpy as np
from hive_dataset import HiveDataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
# Allow imports from the src directory
sys.path.append(
    str(Path(os.path.dirname(os.path.abspath(__file__))).parents[0]))
import config

class Train():

    def __init__(self):
        metadata_column_names = ['device', 'hive number', 'date', 'hive temp', 'hive humidity',
        'hive pressure', 'weather temp', 'weather humidity', 'weather pressure',
        'wind speed', 'gust speed', 'weatherID', 'cloud coverage', 'rain',
        'lat', 'long', 'file name', 'queen presence', 'queen acceptance',
        'frames', 'target', 'time', 'queen status']
        metadata = np.load(config.PROCESSED_METADATA_FILE, allow_pickle=True)
        metadata_df = pd.DataFrame(metadata, columns=metadata_column_names)

        # TODO: Remove
        # metadata_df = metadata_df.head(2000)

        # Train, test, val split
        metadata_train, metadata_test = train_test_split(metadata_df, train_size=0.7, shuffle=True, random_state=42)
        metadata_val, metadata_test = train_test_split(metadata_test, train_size=0.5, shuffle=True, random_state=42)

        # TODO: Remove or make more flexible
        # from torchvision.models.densenet import DenseNet121_Weights
        # transforms_dense_net = DenseNet121_Weights.DEFAULT.transforms()

        # dataset = HiveDataset(metadata_path=config.PROCESSED_METADATA_FILE, processed_data_path=config.PROCESSED_DATA_PATH, target_feature=config.TARGET_FEATURE)
        train_dataset = HiveDataset(metadata_df=metadata_train, processed_data_path=config.NORMALIZED_MEL_SPEC_PATH, target_feature=config.TARGET_FEATURE)
        self.training_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
        val_dataset = HiveDataset(metadata_df=metadata_val, processed_data_path=config.NORMALIZED_MEL_SPEC_PATH, target_feature=config.TARGET_FEATURE)
        self.validation_dataloader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=True)
        test_dataset = HiveDataset(metadata_df=metadata_test, processed_data_path=config.NORMALIZED_MEL_SPEC_PATH, target_feature=config.TARGET_FEATURE)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)

        self.model = CNNModel()
        # self.model = DenseNetTransferModel()

        if torch.cuda.is_available():
            self.model.cuda()
        print(summary(self.model))

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

    def train_one_epoch(self, epoch_index, tb_writer, device="cpu"):
        running_loss = 0.
        avg_loss = 0.

        device = torch.device(device)

        for i, data in enumerate(tqdm(self.training_dataloader)):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # print(outputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            # TODO: Check if this actually updates the weights!
            self.optimizer.step()
            # print(list(self.model.parameters()))

            # for name, param in self.model.named_parameters():
            #     print(name, param.grad)

            # Gather data and report
            running_loss += loss.item()
            avg_loss += loss.item()
            if (i + 1) % self.training_dataloader.batch_size == 0:
                last_loss = running_loss / self.training_dataloader.batch_size  # loss per batch
                # print(f"  batch {i +1} loss: {last_loss:.3f}")
                tb_x = epoch_index * len(self.training_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return avg_loss / (i + 1)


    # Initializing in a separate cell so we can easily add more epochs to the same run
    def train_cnn(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/cnn_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = 3

        best_vloss = np.inf

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training will happen on {device}.")

        for epoch in range(EPOCHS):
            print(f"EPOCH {epoch_number + 1}")

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss_train = self.train_one_epoch(epoch_number, writer, device)

            # TODO: Move this to a separate function
            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_dataloader):
                    if i > 10:
                        break
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print(f"LOSS train {avg_loss_train:.3f} valid {avg_vloss:.3f}")

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss_train, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = os.path.join(config.MODEL_INTERIM_PATH, f"model_{timestamp}_{epoch_number}.pt")
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1


# trainer = Train()
# trainer.train_cnn()
