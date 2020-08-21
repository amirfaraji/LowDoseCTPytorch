# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from dival import get_standard_dataset
from dival.reconstructors.odl_reconstructors import FBPReconstructor

from lodopab_challenge.challenge_set import generator, NUM_IMAGES
from lodopab_challenge.submission import save_reconstruction, pack_submission

# define reconstructor
dataset = get_standard_dataset('lodopab')
reconstructor = FBPReconstructor(dataset.get_ray_trafo())
reconstructor.hyper_params['filter_type'] = 'Hann'
reconstructor.hyper_params['frequency_scaling'] = 0.641

# reconstruct and create submission
OUTPUT_PATH = '/localdata/lodopab_challenge_fbp'
os.makedirs(OUTPUT_PATH, exist_ok=True)
for i, obs in enumerate(tqdm(generator(), total=NUM_IMAGES)):
    reco = reconstructor.reconstruct(obs)
    save_reconstruction(OUTPUT_PATH, i, reco)

pack_submission(OUTPUT_PATH)
