import os
import h5py
import numpy as np

from glob import glob
from utils.RayTransform import RayTransform
from utils.NoiseOperator import NoiseOperator

RT = RayTransform()
NO = NoiseOperator(RT.ray_trafo, resize_shape=RT._IMAGE_SHAPE)

in_dir = os.path.join(os.path.dirname(__file__),'./data/ground_truth_train/')
out_dir = os.path.join(os.path.dirname(__file__),'./data/observation_train/')

in_files = sorted(glob(f'{in_dir}/*.hdf5'))

for i, in_file in enumerate(in_files):

    print(f'Processing set {i+1}/{len(in_files)}')

    with h5py.File(in_file, 'r') as f:
        data = np.asarray(f['data'])

    out_data = np.zeros((data.shape))
    for j in range(len(data)):
        noise_img = NO.apply_noise(data[j,:,:])
        out_data[j,:,:] = np.asarray(RT.fbp(noise_img))

    with h5py.File(f'{out_dir}/observation_train_{str(i).zfill(3)}.hdf5', 'w') as data_file:
        data_file.create_dataset('data', data=out_data)

