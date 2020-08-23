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

def inference():
    net = BiggerUnetPlusPlus(in_channel=1, num_classes=1)
    net.to(device=device, dtype=torch.float)

    # PATH = "./best_model_wts.pt"
    # PATH = os.path.join(os.path.dirname(__file__),"./weights/BestWeights/BiggerUnet_best_model_wts.pt")
    PATH = os.path.join(os.path.dirname(__file__),"./weights/BestWeights/BiggerUnetplusplusbest_model_wts.pt") # Best SSIM: 0.8431391701528878, PSNR: 34.87763746426393
    batch_size = 1

    net.load_state_dict(torch.load(PATH))


    train_observation_filename = os.path.join(os.path.dirname(__file__),"./data/observation_train")
    train_ground_truth_filename = os.path.join(os.path.dirname(__file__),"./data/ground_truth_train")

    val_observation_filename  = os.path.join(os.path.dirname(__file__),"./data/observation_validation")
    val_ground_truth_filename = os.path.join(os.path.dirname(__file__),"./data/ground_truth_validation")

    test_observation_filename  = os.path.join(os.path.dirname(__file__),"./data/observation_test")
    test_ground_truth_filename = os.path.join(os.path.dirname(__file__),"./data/ground_truth_test")

    client = ReconstructionModelClient()

    RT = RayTransform()

    test = LowDoseCTDataset(
        observation_dir    = test_observation_filename,
        ground_truth_dir   = test_ground_truth_filename,
        fbp_op             = RT.fbp,
        phase              = 'test',
        transform          = None,
        target_transform   = None,
        resize             = None,
        preprocessing_flag = False,
    )

    test_dataloader = DataLoader(test, batch_size=1,
                            shuffle=False, num_workers=12)

    # img = test.__getitem__(1)
    # gt = img['gt']
    # img = img['observation']
    # out = client.single_predict(net, torch.unsqueeze(img,0), device)
    # ssim_val = SSIM(out[0,0,:,:], gt[0,:,:])
    # psnr_val = PSNR(out[0,0,:,:], gt[0,:,:])
    # print(ssim_val, psnr_val)
    # visualize(img[0,:,:], gt[0,:,:], out[0,0,:,:])
    evaluation(client, net, test_dataloader, device)
    # predict_on_smaller_images_upscaled_after(client, net, full_size_test_dataloader, half_size_test_dataloader, device)
    

def evaluation(client, model, loader, device):
    
    ssim_val, psnr_val = client.batch_evaluation(model, loader, device)
    print(f'SSIM: {ssim_val}, PSNR: {psnr_val}')

def predict_on_smaller_images_upscaled_after(client, model, full_size_loader, half_size_loader, device):

    outputs = client.predict(model, half_size_loader, device)
    ssim_val, psnr_val = [0]*len(outputs), [0]*len(outputs)

    bar = Bar(f'Processing test set', max=len(full_size_loader))
    for i, batch in enumerate(full_size_loader):

        ground_truths = batch['gt']
        if not isinstance(ground_truths, np.ndarray):
            ground_truths = ground_truths.detach().cpu().numpy() 
        ground_truths = np.squeeze(np.squeeze(ground_truths))
        full_size_loader.dataset.resize = (362,362)
        reshaped_preds = full_size_loader.dataset.resize_image(outputs[i,:,:])

        bar.next()

        ssim_val[i] = SSIM(reshaped_preds, ground_truths)
        psnr_val[i] = PSNR(reshaped_preds, ground_truths)

    bar.finish()
    print(f'SSIM: {np.mean(ssim_val)}, PSNR: {np.mean(psnr_val)}')

if __name__ == '__main__':
    inference()