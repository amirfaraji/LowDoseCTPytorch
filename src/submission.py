import os
import numpy as np
import torch

from client.ModelClient import ReconstructionModelClient, Bar
from model.Models import UNet, BiggerUnet, BiggerUnetPlusPlus
from torch.utils.data import DataLoader
from utils.DataHandler import ChallengeDataset
from utils.RayTransform import RayTransform
from utils.Visualize import *
from dival.measure import SSIM, PSNR

from libs.lodopab_challenge.submission import save_reconstruction, pack_submission



torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference():
    net = BiggerUnetPlusPlus(in_channel=1, num_classes=1)
    net.to(device=device)

    # PATH = os.path.join(os.path.dirname(__file__),"./weights/NoTransformsBestWeightsAt180-180/best_model_wts.pt")
    # PATH = os.path.join(os.path.dirname(__file__),"./weights/BiggerNetworkTransformMSESSIM/best_model_wts.pt")
    # PATH = os.path.join(os.path.dirname(__file__),"./weights/BestWeights/BiggerUnet_best_model_wts.pt")
    PATH = "./best_model_wts.pt"
    batch_size = 1

    net.load_state_dict(torch.load(PATH))


    challenge_observation_filename = os.path.join(os.path.dirname(__file__), "./data/observation_challenge")

    client = ReconstructionModelClient()

    RT = RayTransform()

    challenge = ChallengeDataset(
        observation_dir    = challenge_observation_filename,
        fbp_op             = RT.fbp,
        transform          = None,
        target_transform   = None,
        resize             = None,
        preprocessing_flag = False,
    )

    challenge_dataloader = DataLoader(
        challenge,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    
    OUTPUT_PATH = './src/data/challenge_reconstruction'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    bar = Bar(f'Processing Challenge set', max=len(challenge_dataloader), suffix='%(index)d / %(max)d - %(eta)d secs')

    net.eval()
    for i, batch in enumerate(challenge_dataloader):
        inputs = batch['observation']
        inputs = inputs.to(device, dtype=torch.float)
        outputs = net(inputs)
        bar.next()
        # visualize(inputs[0,0,:,:].detach().cpu().numpy(), outputs[0,0,:,:].detach().cpu().numpy())
        save_reconstruction(OUTPUT_PATH, i, outputs[0, 0, :, :].detach().cpu().numpy())
    bar.finish()
    pack_submission(OUTPUT_PATH)
    


if __name__ == '__main__':
    inference()
