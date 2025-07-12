import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import copy
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
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
VAL_IMG_DIR = "new_data_HAM/test/image/"
VAL_MASK_DIR = "new_data_HAM/test/mask/"

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

# Federated Functions
def train_client(model, loader, optimizer, loss_fn, scaler):
    model.train()
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        data = data.to(DEVICE, dtype=torch.float32)
        targets = targets.float().unsqueeze(1).to(DEVICE)

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


def main():
    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()

    client_loaders = []
    for i in range(NUM_CLIENTS):
        train_loader, _ = get_loaders(
            TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
            BATCH_SIZE, train_transform, val_transform,
            NUM_WORKERS, PIN_MEMORY
        )
        client_loaders.append(train_loader)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), global_model)

    for round in range(ROUNDS):
        print(f"Round {round+1}/{ROUNDS}")
        client_models = []
        for i, loader in enumerate(client_loaders):
            print(f"Client {i+1}")
            client_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(client_model.parameters(), lr=LEARNING_RATE)
            scaler = torch.cuda.amp.GradScaler()
            client_state_dict = train_client(client_model, loader, optimizer, loss_fn, scaler)
            client_models.append(client_state_dict)

        # Aggregate Models
        federated_averaging(global_model, client_models)
        check_accuracy(loader, global_model, device=DEVICE)
        save_checkpoint({"state_dict": global_model.state_dict()})
        save_predictions_as_imgs(loader, global_model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()
