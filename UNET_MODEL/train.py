import os
import time
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utlis import seeding, create_dir, epoch_time
import logging

# Initialize the logger
logger = logging.getLogger(__name__)

def train(model, loader, optimizer, loss_fn, device):
    """ Function to train the model for one epoch """
    epoch_loss = 0.0
    model.train()

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad() 
        y_pred = model(x) 
        loss = loss_fn(y_pred, y)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        epoch_loss += loss.item()  

    epoch_loss = epoch_loss / len(loader)  # Calculate average loss
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    """ Function to evaluate the model on validation set """
    epoch_loss = 0.0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)  # Forward pass
            loss = loss_fn(y_pred, y)  # Calculate the loss
            epoch_loss += loss.item()  # Accumulate the loss

    epoch_loss = epoch_loss / len(loader)  # Calculate average loss
    return epoch_loss


if __name__ == '__main__':
    # Seeding for reproducibility
    seeding(42)
    logger.info("Seeding set to 42 for reproducibility.")

    # Create necessary directories
    create_dir("files")
    logger.info("Directory 'files' created for storing model checkpoints.")

    # Load the dataset file paths
    train_x = sorted(glob("./new_data/train/image/*"))[:200]
    train_y = sorted(glob("./new_data/train/mask/*"))[:200]
    valid_x = sorted(glob("./new_data/val/image/*"))[:20]
    valid_y = sorted(glob("./new_data/val/mask/*"))[:20]

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}"
    logger.info(data_str)

    # Hyperparameters
    H, W = 512, 512  # Image dimensions
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    logger.info(f"Hyperparameters set: Batch size: {batch_size}, Epochs: {num_epochs}, Learning rate: {lr}")

    # Create Dataset and DataLoader instances
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info("DataLoader instances created for training and validation datasets.")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    # Training the model
    best_valid_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training and evaluation
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        # Save the best model
        if valid_loss < best_valid_loss:
            logger.info(f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint to {checkpoint_path}")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        # Log epoch time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Logging the results of the epoch
        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s"
        data_str += f"\n\tTrain Loss: {train_loss:.3f}\n\tVal. Loss: {valid_loss:.3f}\n"
        logger.info(data_str)
