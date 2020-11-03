import numpy as np
from PIL import Image
import glob
import torch
from torch.utils.data.dataset import Dataset

'''
Pytorch uses datasets and has a very handy way of creatig dataloaders in your main.py
Make sure you read enough documentation.
'''


class Data(Dataset):
    def __init__(self, data_dir):
        # gets the data from the directory
        self.image_list = glob.glob(data_dir + '*')
        # calculates the length of image_list
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        single_image_path = self.image_list[index]
        # Open image
        image = Image.open(single_image_path)

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        image_np = np.reshape(np.asarray(image), (-1, 3, 32, 32)) / 255
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension

        image_np = np.expand_dims(image_np, 0)
        '''
        #TODO: Convert your numpy to a tensor and get the labels
        '''
        image_tensor = torch.from_numpy(image_np).float()
        class_indicator_location = single_image_path.rfind('_c')
        label = int(single_image_path[class_indicator_location + 2:class_indicator_location + 3])
        return image_tensor, label

    def __len__(self):
        return self.data_len
