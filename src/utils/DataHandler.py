import os
import h5py
import time
import torch
import random
import numpy as np

from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from .Preprocessing import Preprocess
from .RayTransform import RayTransform
from .NoiseOperator import NoiseOperator


class LowDoseCTDataset(Dataset):

    def __init__(
            self, 
            observation_dir: str, 
            ground_truth_dir: str, 
            fbp_op, 
            phase: str='train', 
            transform=None, 
            target_transform=None, 
            resize: tuple=None,
            preprocessing_flag: bool=True
        ):
        
        self.observation_dir = observation_dir
        self.ground_truth_dir = ground_truth_dir
        self.fbp = fbp_op
        self.phase = phase

        self.observation_filenames = sorted(glob(f'{self.observation_dir}/observation_{self.phase}_*.hdf5'))
        self.ground_truth_filenames = sorted(glob(f'{self.ground_truth_dir}/ground_truth_{self.phase}_*.hdf5'))

        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        self.preprocessing_flag = preprocessing_flag

        self.__dataset_length = self.get_length()
        self.__shape = self.__getitem__(0)['gt'].shape

    def __len__(self):
        return self.__dataset_length

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        observation_filename =  self.observation_filenames[idx // self.num_images_per_file]
        ground_truth_filename =  self.ground_truth_filenames[idx // self.num_images_per_file]
        data_idx = idx % self.num_images_per_file

        observation_data = self.fbp(self.read_hdf5(observation_filename)[data_idx])
        ground_truth_data = self.read_hdf5(ground_truth_filename)[data_idx]

        if self.preprocessing_flag:
            observation_data = Preprocess.normalize(Preprocess.median_filter(observation_data, size=3), max_val=np.max(observation_data), min_val=np.min(observation_data))

        if self.resize:
            observation_data = self.resize_image(observation_data)
            ground_truth_data = self.resize_image(ground_truth_data)
        
        sample = {'observation': torch.from_numpy(np.expand_dims(observation_data,axis=0)), 'gt': torch.from_numpy(np.expand_dims(ground_truth_data,axis=0))}

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        if self.transform:
            sample['observation'] = self.transform(sample['observation'])
            
        random.seed(seed) # apply this seed to target tranfsorms
        if self.target_transform:
            sample['gt'] = self.target_transform(sample['gt'])

        return sample
    
    @property
    def shape(self):
        return self.__shape

    def get_length(self, which_files: str='observation') -> int:
        if which_files == 'ground_truth':
            files = self.ground_truth_filenames
        else:
            files = self.observation_filenames
        num_files = len(files)
        self.num_images_per_file = len(self.read_hdf5(files[0]))
        return (num_files - 1)*self.num_images_per_file  + len(self.read_hdf5(files[-1]))

    def read_hdf5(self, filename: str) -> np.array:

        with h5py.File(filename, "r") as f:
            return np.asarray(f['data'])

    def resize_image(self, image: Image) -> np.array:
        img = Image.fromarray(image)
        return np.asarray(img.resize(size=self.resize, resample=Image.BICUBIC))


class TrainDataset(LowDoseCTDataset):

    def __init__(
            self, 
            ground_truth_dir: str, 
            ray_transform_class: RayTransform, 
            transform=None, 
            target_transform=None, 
            resize: tuple=None, 
            preprocessing_flag: bool=True
        ):
        
        self.ground_truth_dir = ground_truth_dir
        self.phase = 'train'

        self.ground_truth_filenames = sorted(glob(f'{self.ground_truth_dir}/ground_truth_{self.phase}_*.hdf5'))

        self.ray_transform = ray_transform_class.ray_trafo
        self.fbp = ray_transform_class.fbp
        self.resize_shape = ray_transform_class._IMAGE_SHAPE
        self.noise_op = NoiseOperator(ray_transform=self.ray_transform, resize_shape=self.resize_shape)

        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        self.preprocessing_flag = preprocessing_flag

        self.__dataset_length = self.get_length(which_files='ground_truth')
        self.__shape = self.__getitem__(0)['gt'].shape

    def __len__(self):
        return self.__dataset_length

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ground_truth_filename =  self.ground_truth_filenames[idx // self.num_images_per_file]
        data_idx = idx % self.num_images_per_file

        ground_truth_data = self.read_hdf5(ground_truth_filename)[data_idx]
        observation_data = np.asarray(self.fbp(self.generate_noisy_observation(ground_truth_data)))

        if self.resize:
            observation_data = self.resize_image(observation_data)
            ground_truth_data = self.resize_image(ground_truth_data)

        if self.preprocessing_flag:
            observation_data = Preprocess.normalize(Preprocess.median_filter(observation_data, size=3), max_val=np.max(observation_data), min_val=np.min(observation_data))

        sample = {'observation': torch.from_numpy(np.expand_dims(observation_data,axis=0)), 'gt': torch.from_numpy(np.expand_dims(ground_truth_data,axis=0))}

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        if self.transform:
            sample['observation'] = self.transform(sample['observation'])
            
        random.seed(seed) # apply this seed to target tranfsorms
        if self.target_transform:
            sample['gt'] = self.target_transform(sample['gt'])

        return sample

    @property
    def shape(self):
        return self.__shape

    def generate_noisy_observation(self, data: np.array) -> np.array:
        return self.noise_op.apply_noise(data)


class ChallengeDataset(LowDoseCTDataset):

    def __init__(
            self, 
            observation_dir: str, 
            fbp_op, 
            transform=None, 
            target_transform=None, 
            resize: tuple=None, 
            preprocessing_flag: bool=True
        ):
        
        self.observation_dir = observation_dir
        self.phase = 'challenge'
        self.fbp = fbp_op
        
        self.observation_filenames = sorted(glob(f'{self.observation_dir}/observation_{self.phase}_*.hdf5'))


        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        self.preprocessing_flag = preprocessing_flag

        self.__dataset_length = self.get_length(which_files='observation')
        self.__shape = self.__getitem__(0)['observation'].shape

    def __len__(self):
        return self.__dataset_length

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        observation_filename =  self.observation_filenames[idx // self.num_images_per_file]
        data_idx = idx % self.num_images_per_file

        observation_data = self.fbp(self.read_hdf5(observation_filename)[data_idx])

        if self.resize:
            observation_data = self.resize_image(observation_data)

        if self.preprocessing_flag:
            observation_data = Preprocess.normalize(Preprocess.median_filter(observation_data, size=3), max_val=np.max(observation_data), min_val=np.min(observation_data))

        sample = {'observation': torch.from_numpy(np.expand_dims(observation_data,axis=0))}

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        if self.transform:
            sample['observation'] = self.transform(sample['observation'])
            
        random.seed(seed) # apply this seed to target tranfsorms
        if self.target_transform:
            sample['gt'] = self.target_transform(sample['gt'])

        return sample

    @property
    def shape(self):
        return self.__shape

