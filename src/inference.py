import os
import argparse
import numpy as np
import torch

from client.ModelClient import ReconstructionModelClient, Bar
from model.Models import UNet, BiggerUnet, UnetPlusPlus, BiggerUnetPlusPlus
from torch.utils.data import DataLoader
from utils.DataHandler import LowDoseCTDataset
from utils.RayTransform import RayTransform
from utils.Visualize import *
from dival.measure import SSIM, PSNR



torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description='Inference for Low Dose CT Scans')

    parser.add_argument('--test-batch-size', type=int, default=1, metavar='B',
                        help='input batch size for test (default: 1)')
    parser.add_argument('--workers', type=int, default=12,
                        help='number of workers for data loader (default: 12)')
    parser.add_argument('--path', type=str, default='./best_model_wts.pt',
                        help='Path for loading Model')
    args = parser.parse_args()

    return args


def inference():

    args = get_args()

    net = BiggerUnetPlusPlus(in_channel=1, num_classes=1)
    net.to(device=device, dtype=torch.float)

    PATH = args.path  # "./best_model_wts.pt" # Best SSIM: 0.8431391701528878, PSNR: 34.87763746426393
    # PATH = os.path.join(os.path.dirname(__file__),"./weights/UnetMSESSSIM/best_model_wts.pt") # SSIM: 0.8012296540502425, PSNR: 32.05215252747701
    # PATH = os.path.join(os.path.dirname(__file__),"./weights/BestWeights/SmallerUnet_best_model_wts.pt") # SSIM: 0.8022275263184017, PSNR: 32.060868586650976
    # PATH = os.path.join(os.path.dirname(__file__),"./weights/BestWeights/BiggerUnet_best_model_wts.pt") # SSIM: 0.8042886567222887, PSNR: 32.670939523169096
    # PATH = os.path.join(os.path.dirname(__file__),"./weights/BestWeights/BiggerUnetplusplusbest_model_wts.pt") # Best SSIM: 0.8431391701528878, PSNR: 34.87763746426393
    net.load_state_dict(torch.load(PATH))
    
    batch_size = args.test_batch_size  # 1
    num_workers = args.workers  # 12
    preprocessing_flag = False
    vis_test_flag = False

    test_observation_filename  = os.path.join(os.path.dirname(__file__),"./data/observation_test")
    test_ground_truth_filename = os.path.join(os.path.dirname(__file__),"./data/ground_truth_test")

    client = ReconstructionModelClient()

    RT = RayTransform()

    test = LowDoseCTDataset(
        observation_dir    = test_observation_filename,
        ground_truth_dir   = test_ground_truth_filename,
        fbp_op             = RT.fbp,
        phase              = 'test',
        transfo_flag       = False,
        resize             = None,
        preprocessing_flag = preprocessing_flag,
    )

    test_dataloader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    
    evaluation(client, net, test_dataloader, device)
    
    if vis_test_flag:
        visualize_test_img(client, net, device, test)
    

def evaluation(client, model, loader, device):
    
    ssim_val, psnr_val = client.batch_evaluation(model, loader, device)
    print(f'SSIM: {ssim_val}, PSNR: {psnr_val}')


def visualize_test_img(client, model, device, test_data, idx: int = 0):

    img = test_data.__getitem__(idx)
    gt = img['gt']
    img = img['observation']

    out = client.single_predict(model, torch.unsqueeze(img,0), device)

    ssim_val = SSIM(out[0,0,:,:], gt[0,:,:])
    psnr_val = PSNR(out[0,0,:,:], gt[0,:,:])

    print(ssim_val, psnr_val)
    visualize(img[0,:,:], gt[0,:,:], out[0,0,:,:])


if __name__ == '__main__':
    inference()
