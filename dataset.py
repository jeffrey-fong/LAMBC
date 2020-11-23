import torch
from torch.utils.data import Dataset
import numpy as np


class ImageNetDataset(Dataset):

    def __init__(self, train=True):
        self.train = train
        self.load_file()

    def load_file(self):

        data = None
        label = None

        if self.train:
            for i in range(1):
                data_path = f"../image_datasets/Imagenet64_train_part1_npz/train_data_batch_{i + 1}.npz"
                data_seg = np.load(data_path)
                image_seg = np.reshape(data_seg['data'], (-1, 3, 64, 64)).astype(np.float32)
                label_seg = data_seg['labels']

                data = image_seg if data is None else np.concatenate((data, image_seg), axis=0)
                label = label_seg if label is None else np.concatenate((label, label_seg), axis=0)

        else:
            data_path = f"../image_datasets/Imagenet64_val_npz/val_data.npz"
            data_seg = np.load(data_path)
            data = np.reshape(data_seg['data'], (-1, 3, 64, 64)).astype(np.float32)
            label = data_seg['labels']

        self.data = data
        self.label = label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):

        return self.data[idx], self.label[idx] - 1  # convert 1-1000 to 0-999
