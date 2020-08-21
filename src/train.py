import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from client.ModelClient import ReconstructionModelClient
from model.Models import UNet, BiggerUnet, UnetPlusPlus
from torch.utils.data import DataLoader
from utils.DataHandler import TrainDataset, LowDoseCTDataset
from utils.RayTransform import RayTransform
from utils.Preprocessing import Preprocess
from utils.Metrics import SSIMLoss, MSSSIMLoss, MSESSIMLoss, MAESSIMLoss, MixMSEMAELoss
from utils.Visualize import *



torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Low Dose CT Scans')

    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--patience', type=int, default=15, metavar='S',
                        help='random seed (default: 15)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    return args

def no_fbp(x: np.array) -> np.array:
    return x


def train():
    net = UnetPlusPlus(in_channel=1, num_classes=1)
    net.to(device=device, dtype=torch.float)
    
    epochs=7
    batch_size = 4
    lr=1e-4
    patience=3

    criterion = MSESSIMLoss() # MSESSIMLoss() MAESSIMLoss() MSSSIMLoss() SSIMLoss() nn.MSELoss() MixMSEMAELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.num_classes > 1 else 'max', patience=patience)

    train_observation_filename = os.path.join(os.path.dirname(__file__),"./data/observation_train")
    train_ground_truth_filename = os.path.join(os.path.dirname(__file__),"./data/ground_truth_train")

    val_observation_filename  = os.path.join(os.path.dirname(__file__),"./data/observation_validation")
    val_ground_truth_filename = os.path.join(os.path.dirname(__file__),"./data/ground_truth_validation")

    test_observation_filename  = os.path.join(os.path.dirname(__file__),"./data/observation_test")
    test_ground_truth_filename = os.path.join(os.path.dirname(__file__),"./data/ground_truth_test")

    client = ReconstructionModelClient()

    RT = RayTransform()

    

    data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    target_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    train = LowDoseCTDataset(
        observation_dir    = train_observation_filename,
        ground_truth_dir   = train_ground_truth_filename,
        fbp_op             = no_fbp,
        phase              = 'train',
        transform          = None,
        target_transform   = None,        
        resize             = None,
        preprocessing_flag = True,
    )

    # train = TrainDataset(
    #     ground_truth_dir    = train_ground_truth_filename,
    #     ray_transform_class = RT,
    #     transform           = None,
    #     target_transform    = None,
    #     resize              = None,
    #     preprocessing_flag  = True,
    # )

    validation = LowDoseCTDataset(
        observation_dir    = val_observation_filename,
        ground_truth_dir   = val_ground_truth_filename,
        fbp_op             = RT.fbp,
        phase              = 'validation',
        transform          = None,
        target_transform   = None,        
        resize             = None,
        preprocessing_flag = True,
    )



    train_dataloader = DataLoader(train, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    validation_dataloader = DataLoader(validation, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    
    best_model = client.train(net, train_dataloader, validation_dataloader, criterion, optimizer, None, device, epochs)

    plot_history(client.history)

if __name__ == '__main__':
    train()
    