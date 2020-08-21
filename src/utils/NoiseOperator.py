import numpy as np
from skimage.transform import resize
import multiprocessing




class NoiseOperator():
    
    def __init__(self, 
                ray_transform,
                resize_shape=None,
                seed_flag = False,
                seed = None
                ):

        self.ray_transform = ray_transform
        self.resize_shape = resize_shape
        
        self.PHOTONS_PER_PIXEL = 4096
        self.noise_dict = {
            'poisson' : self.poisson_op,
            'gaussian': self.poisson_op,
            'salt'    : self.poisson_op,
            'pepper'  : self.poisson_op,
            's&p'     : self.poisson_op,
        }

        if seed_flag:
            np.random.RandomState(seed)

        self.__set_linear_attenuation_coef()
    
    def __set_linear_attenuation_coef(self):

        self._MU_WATER = 20
        self._MU_AIR = 0.02
        self._MU_MAX = 3071 * (self._MU_WATER - self._MU_AIR) / 1000 + self._MU_WATER


    def forward_op(self, image: np.array) -> np.array:

        if self.resize_shape is None:
            self.resize_shape = image.shape

        image_resized = resize(image * self._MU_MAX, self.resize_shape, order=1)
        data = self.ray_transform(image_resized).asarray()

        data *= (-1)
        np.exp(data, out=data)
        data *= self.PHOTONS_PER_PIXEL
        return data

    def apply_noise(self, data: np.array, noise_types ='poisson') -> np.array:
        
        if not isinstance(noise_types,list):
            noise_types = [noise_types]

        for noise_type in noise_types:
            if noise_type not in self.noise_dict.keys():
                print(f"Warning: No operation for {noise_type} was found.")
            else:
                data = self.noise_dict[noise_type](data)
        
        return data

    def poisson(self, data: np.array) -> np.array:
        return np.random.poisson(data)

    def poisson_op(self, data: np.array) -> np.array:

        noise = self.poisson(self.forward_op(data)) / self.PHOTONS_PER_PIXEL
        np.maximum(0.1 / self.PHOTONS_PER_PIXEL, noise, out=noise)
        np.log(noise, out=noise)
        noise /= (-self._MU_MAX)
        
        return noise

    def gaussian(self):
        raise NotImplementedError()

    def salt(self):
        raise NotImplementedError()

    def pepper(self):
        raise NotImplementedError()

    def salt_and_pepper(self):
        raise NotImplementedError()

    
