import numpy as np
from PIL import Image
import glob
import torch
from torch.utils.data.dataset import Dataset
import _pickle as cp

'''
Pytorch uses datasets and has a very handy way of creatig dataloaders in your main.py
Make sure you read enough documentation.
'''


class Data(Dataset):
    def __init__(self, data_dir, batch_size=10000):
        # gets the data from the directory
        self.image_batches = glob.glob(data_dir + '*')
        self.batch_data = len(self.image_batches) * [None]
        self.file_batch_size = batch_size
        # calculates the length of image_list
        self.data_len = batch_size * len(self.image_batches)

    def __getitem__(self, index):
        if self.batch_data[index // self.file_batch_size] is None:
            batch_path = self.image_batches[index // self.file_batch_size]
            self.batch_data[index // self.file_batch_size] = cp.load(open(batch_path, 'rb'))
        batch = self.batch_data[index // self.file_batch_size]
        array = batch['data']
        labels = batch['labels']
        image_np = np.reshape(np.asarray(array), (-1, 3, 32, 32)) / 255
        image_np = image_np[index % self.file_batch_size, :, :, :]
        label = labels[index % self.file_batch_size]
        image_tensor = torch.from_numpy(image_np).float()
        return image_tensor, label

    def __len__(self):
        return self.data_len
