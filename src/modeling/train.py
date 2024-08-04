
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchinfo import summary
from datetime import datetime
from tqdm import tqdm
from model import CNNModel
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

    def __init__(self, model=None, loss_fn=None):
        # Load metadata
        metadata_column_names = ['sample_name', "label", "hive number", "segment",]
        metadata = np.load(config.PROCESSED_METADATA_FILE_SEGMENTED, allow_pickle=True)
        metadata_df = pd.DataFrame(metadata, columns=metadata_column_names)
        metadata_df = metadata_df.astype({'label': 'int32', 'hive number': 'int32'})

        # Train, test, val split
        # Split by sample_name to avoid data leakage
        unique_samples = metadata_df["sample_name"].unique()
        self.train_samples, eval_samples = train_test_split(unique_samples, train_size=0.7, shuffle=True, random_state=42)
        self.val_samples, self.test_samples = train_test_split(eval_samples, train_size=0.5, shuffle=True, random_state=42)
    
        metadata_train = metadata_df[metadata_df["sample_name"].isin(self.train_samples)]
        metadata_val = metadata_df[metadata_df["sample_name"].isin(self.val_samples)]
        metadata_test = metadata_df[metadata_df["sample_name"].isin(self.test_samples)]

        train_dataset = HiveDataset(metadata_df=metadata_train, processed_data_path=config.NORMALIZED_MEL_SPEC_PATH, target_feature=config.TARGET_FEATURE)
        self.training_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        val_dataset = HiveDataset(metadata_df=metadata_val, processed_data_path=config.NORMALIZED_MEL_SPEC_PATH, target_feature=config.TARGET_FEATURE)
        self.validation_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)
        test_dataset = HiveDataset(metadata_df=metadata_test, processed_data_path=config.NORMALIZED_MEL_SPEC_PATH, target_feature=config.TARGET_FEATURE)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

        # Initialize model
        self.model = model

        # Send model to GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
        print(summary(self.model))

        # Initialize loss function and optimizer
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)


    def train_one_epoch(self, epoch_index, tb_writer, device="cpu"):
        running_loss = 0.
        avg_loss = 0.

        device = torch.device(device)

        for i, data in enumerate(tqdm(self.training_dataloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)
            outputs = outputs.squeeze(1)

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
                # print(f"  batch {i +1} loss: {last_loss:.3f}")
                tb_x = epoch_index * len(self.training_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return avg_loss / (i + 1)


    def evaluate(self, device="cpu"):
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
                voutputs = voutputs.squeeze(1)
                vloss = self.loss_fn(voutputs, vlabels)
                running_vloss += vloss
        return running_vloss / (i + 1)


    def train_cnn(self, epochs):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/cnn_trainer_{}'.format(timestamp))
        epoch_number = 0

        best_vloss = np.inf

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training will happen on {device}.")

        if not os.path.exists(config.MODEL_INTERIM_PATH):
            os.makedirs(config.MODEL_INTERIM_PATH)

        for epoch in range(epochs):
            print(f"EPOCH {epoch_number + 1}")

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss_train = self.train_one_epoch(epoch_number, writer, device)

            # Evaluate the model on the validation set
            avg_loss_val = self.evaluate(device)            
            print(f"LOSS train: {avg_loss_train:.3f}, val: {avg_loss_val:.3f}")

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss_train, 'Validation' : avg_loss_val },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_loss_val < best_vloss:
                best_vloss = avg_loss_val
                checkpoint_path = os.path.join(config.MODEL_INTERIM_PATH, f"model_{timestamp}_{epoch_number}_checkpoint.pt")
                model_path = os.path.join(config.MODEL_INTERIM_PATH, f"model_{timestamp}_{epoch_number}.pt")

                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_samples': self.train_samples,
                    'val_samples': self.val_samples,
                    'test_samples': self.test_samples,
                }

                torch.save(checkpoint, checkpoint_path)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
        writer.close()


    def load_model(self, checkpoint_path, model, optimizer):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_samples = checkpoint['train_samples']
        val_samples = checkpoint['val_samples']
        test_samples = checkpoint['test_samples']
        return model, optimizer, checkpoint['epoch'], train_samples, val_samples, test_samples