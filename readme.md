# UNet Segmentation with Supervised Federated Learning using IID-data distribution

This project implements semantic segmentation using a UNet architecture on image data, supporting both supervised learning and federated learning across multiple clients.

## Project Structure

- `model.py`: Defines the UNet model architecture.
- `dataset.py`: Custom dataset loader for images and masks.
- `utils.py`: Utility functions for training, evaluation, and saving checkpoints.
- `train.py`: Standard supervised training script.
- `fed.py`: Federated learning training script, splitting data among clients and aggregating models.
- `readme.md`: Project documentation.
- `Checkpoint.pth`: Dont forget to create an checkpoint pth file where model saves it's last federated round

## Requirements

- Python 3.7+
- PyTorch
- Albumentations
- torchvision
- tqdm
- Pillow
- OpenCV

Install dependencies:

```sh
pip install torch torchvision albumentations tqdm pillow opencv-python
```

Datasets i haved used : there is no need to download whole dataset just download few training and testing data for experimental use (eg. 300 Train set and 20 Test set)

HAM1000 : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

BUSI: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
