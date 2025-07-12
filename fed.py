import torch
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
NUM_CLIENTS = 10
ROUNDS = 5  # Number of federated rounds

TRAIN_IMG_DIR = "new_data_HAM/train/image/"
TRAIN_MASK_DIR = "new_data_HAM/train/mask/"
TEST_IMG_DIR = "new_data_HAM/test/image/"  # Added path for test images
TEST_MASK_DIR = "new_data_HAM/test/mask/"  # Added path for test masks

# Data Transforms
train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])


# Custom Dataset to load images and masks
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, mask_names, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.mask_names = mask_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        # Read image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure mask is float32 and normalize to [0, 1] range if necessary
        mask = mask.astype('float32') / 255.0  # If your masks are binary (0 or 255)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


# Federated Functions
def train_client(model, loader, optimizer, loss_fn, scaler):
    model.train()
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        data = data.to(DEVICE, dtype=torch.float32)
        targets = targets.float().unsqueeze(1).to(DEVICE)  # Ensure targets are float

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return copy.deepcopy(model.state_dict())


def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.mean(
            torch.stack([client[key].float() for client in client_models]), dim=0
        )
    global_model.load_state_dict(global_dict)


def split_dataset(image_dir, mask_dir, num_clients):
    all_images = sorted(os.listdir(image_dir))
    all_masks = sorted(os.listdir(mask_dir))

    num_samples_per_client = len(all_images) // num_clients
    client_datasets = []

    for i in range(num_clients):
        client_images = all_images[i * num_samples_per_client: (i + 1) * num_samples_per_client]
        client_masks = all_masks[i * num_samples_per_client: (i + 1) * num_samples_per_client]

        # Create Dataset and DataLoader for each client
        dataset = SegmentationDataset(
            image_dir, mask_dir, client_images, client_masks, transform=train_transform
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        client_datasets.append(loader)

    return client_datasets


# Function to get loaders for test set as well
def get_loaders(img_dir, mask_dir, batch_size, transform=None):
    image_names = sorted(os.listdir(img_dir))
    mask_names = sorted(os.listdir(mask_dir))

    dataset = SegmentationDataset(img_dir, mask_dir, image_names, mask_names, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return loader


def main():
    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()

    # Split dataset into NUM_CLIENTS partitions
    client_loaders = split_dataset(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, NUM_CLIENTS
    )

    # Load test dataset separately
    test_loader = get_loaders(
        TEST_IMG_DIR, TEST_MASK_DIR, BATCH_SIZE, val_transform
    )  # Assuming get_loaders can also load the test set

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), global_model)

    for round in range(ROUNDS):
        print(f"Round {round + 1}/{ROUNDS}")
        client_models = []

        for i, loader in enumerate(client_loaders):
            print(f"Client {i + 1}")
            client_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(client_model.parameters(), lr=LEARNING_RATE)
            scaler = torch.cuda.amp.GradScaler()
            client_state_dict = train_client(client_model, loader, optimizer, loss_fn, scaler)
            client_models.append(client_state_dict)

        # Aggregate Models
        federated_averaging(global_model, client_models)

        # Evaluate on the TEST set (not the training set)
        check_accuracy(test_loader, global_model, device=DEVICE)

        # Save predictions on the test set (not the training set)
        save_predictions_as_imgs(test_loader, global_model, folder="saved_images1/", device=DEVICE)

        # Save model checkpoint after each round
        save_checkpoint({"state_dict": global_model.state_dict()})


if __name__ == "__main__":
    main()
