import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from client.ModelClient import ReconstructionModelClient
from model.Models import *
from torch.utils.data import DataLoader
from utils.DataHandler import LowDoseCTDataset
from utils.RayTransform import RayTransform
from utils.Preprocessing import Preprocess
from utils.Metrics import SSIMLoss, MSSSIMLoss, MSESSIMLoss, MAESSIMLoss, MixMSEMAELoss
from utils.Visualize import *



torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():

    parser = argparse.ArgumentParser(description='Training Low Dose CT Scans')

    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=8,
                        help='number of epochs to train (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for data loader (default: 4)')
    parser.add_argument('--patience', type=int, default=15,
                        help='random seed (default: 15)')
    parser.add_argument('--path', type=str, default='./best_model_wts.pt',
                        help='Path for loading Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For loading the Model')
    args = parser.parse_args()

    return args


def no_fbp(x: np.array) -> np.array:
    return x


def train():

    args = get_args()

    net = BiggerUnetPlusPlus(in_channel=1, num_classes=1)
    net.to(device=device, dtype=torch.float)

    if args.load_model:
        PATH = args.path
        net.load_state_dict(torch.load(PATH))
    
    batch_size = args.batch_size  # 4
    epochs = args.epochs  # 8
    lr = args.lr  # 1e-5
    num_workers = args.workers  # 4
    patience = args.patience  # 3
    preprocessing_flag = False

    criterion = MSESSIMLoss()  # MSESSIMLoss() MAESSIMLoss() MSSSIMLoss() SSIMLoss() nn.MSELoss() MixMSEMAELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)  # optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.num_classes > 1 else 'max', patience=patience)

    train_observation_filename  = os.path.join(os.path.dirname(__file__), "./data/observation_train")
    train_ground_truth_filename = os.path.join(os.path.dirname(__file__), "./data/ground_truth_train")

    val_observation_filename  = os.path.join(os.path.dirname(__file__), "./data/observation_validation")
    val_ground_truth_filename = os.path.join(os.path.dirname(__file__), "./data/ground_truth_validation")

    test_observation_filename  = os.path.join(os.path.dirname(__file__), "./data/observation_test")
    test_ground_truth_filename = os.path.join(os.path.dirname(__file__), "./data/ground_truth_test")

    client = ReconstructionModelClient()

    RT = RayTransform()

    train = LowDoseCTDataset(
        observation_dir    = train_observation_filename,
        ground_truth_dir   = train_ground_truth_filename,
        fbp_op             = no_fbp,
        phase              = 'train',
        transfo_flag       = True,
        resize             = None,
        preprocessing_flag = preprocessing_flag,
    )

    validation = LowDoseCTDataset(
        observation_dir    = val_observation_filename,
        ground_truth_dir   = val_ground_truth_filename,
        fbp_op             = RT.fbp,
        phase              = 'validation',
        transfo_flag       = False,
        resize             = None,
        preprocessing_flag = preprocessing_flag,
    )


    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    validation_dataloader = DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    best_model = client.train(net, train_dataloader, validation_dataloader, criterion, optimizer, None, device, epochs)

    plot_history(client.history)


if __name__ == '__main__':
    train()
