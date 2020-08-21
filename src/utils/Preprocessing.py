import numpy as np

from scipy import ndimage
from skimage import restoration
from skimage.transform import iradon


class Preprocess():

    @staticmethod
    def normalize(arr: np.array, max_val: int=1, min_val: int=0) -> np.array:
        
        return (max_val - min_val) * ((arr - np.min(arr))/(np.max(arr) - np.min(arr) + 1e-16)) + min_val


    @staticmethod
    def median_filter(observation, size: int=7, normalize: bool=True) -> np.array:

        if normalize:
            observation = Preprocess.normalize(observation)

        return ndimage.median_filter(observation/np.max(observation), size=size)

    @staticmethod
    def richardson_lucy_algorithm(observation, psf: np.array=None, iterations: int=5, normalize: bool=True) -> np.array:

        if normalize:
            observation = Preprocess.normalize(observation)

        if psf is None:
            size = 5
            psf = np.ones((size, size)) / size**2

        return restoration.richardson_lucy(observation, psf, iterations=iterations)