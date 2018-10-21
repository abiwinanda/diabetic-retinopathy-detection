import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets

def seperate_dataset_to_labels_folder(src, labels_csv, label_rule, train_test_split=0.8):
    pass


def create_dataloader(data_dir, batch_size, shuffle=True, n_thread):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size = batch_size, shuffle = shuffle, num_workers = n_thread) for x in ['train', 'val']}

    return dataloaders_dict
