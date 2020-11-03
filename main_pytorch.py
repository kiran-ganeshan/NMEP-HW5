import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader
from data_pytorch import Data
from resnet_pytorch import ResNet
import time
import shutil
import yaml
import argparse

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        criterion(model(input), target).backward()
        optimizer.step()


def validate(val_loader, model, criterion):
    model.eval()
    total_loss = 0
    for i, (input, target) in enumerate(val_loader):
        total_loss += criterion(model(input), target)
    return total_loss / len(val_loader)


def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar',
                    filename2='rotationnetmodelbest.pth.tar'):
    torch.save(state, filename)
    # best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)


def main():
    n_epochs = config["num_epochs"]
    models = [ResNet(block, layer, 4) for block, layer in ____]
    criterion = nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(model.parameters()) for model in models]

    train_dataset = Data('/Users/kiranganeshan/nmep/hw5/data/train/')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = Data('/Users/kiranganeshan/nmep/hw5/data/test/')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    for epoch in range(n_epochs):
        for model, optimizer in zip(models, optimizers):
            train(train_loader, model, criterion, optimizer, epoch)

    best_loss = validate(val_loader, models[0], criterion)
    for model in models:
        loss = validate(val_loader, model, criterion)
        save_checkpoint(model.state_dict(), loss < best_loss)
        best_loss = min(loss, best_loss)


if __name__ == "__main__":
    main()
