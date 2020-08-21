# -*- coding: utf-8 -*-
import unittest
import numpy as np
from dival import get_standard_dataset

# NOTE: in order to run this test there are some requirements:
# - dival library installed
# - public lodopab dataset downloaded and configured with dival
# - lodopab challenge set observations downloaded (path can be adjusted below)

from lodopab_challenge.challenge_set import (
    config, NUM_ANGLES, NUM_DET_PIXELS, MU_MAX, get_observation,
    get_observations, generator, transform_to_pre_log,
    replace_min_photon_count)

# config['data_path'] = '/localdata/lodopab_challenge_set'

lodopab = get_standard_dataset('lodopab', impl='skimage')

class TestGetObservation(unittest.TestCase):
    def test(self):
        n = 3
        for i, obs2 in zip(range(n), generator()):
            obs = get_observation(i)
            self.assertEqual(obs.shape, (NUM_ANGLES, NUM_DET_PIXELS))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.all(obs == obs2))
            obs = np.zeros((NUM_ANGLES, NUM_DET_PIXELS), dtype=np.float32)
            obs_ = get_observation(i, out=obs)
            self.assertIs(obs_, obs)
            self.assertTrue(np.all(obs == obs2))
            obs = lodopab.space[0].zero()
            obs_ = get_observation(i, out=obs)
            self.assertIs(obs_, obs)
            self.assertTrue(np.all(obs == obs2))
            with self.assertRaises(IndexError):
                get_observation(3678)
            with self.assertRaises(IndexError):
                get_observation(-3679)

class TestGetObservations(unittest.TestCase):
    def test(self):
        n = 3
        obs = get_observations(range(n))
        self.assertEqual(obs.shape, (n, NUM_ANGLES, NUM_DET_PIXELS))
        self.assertEqual(obs.dtype, np.float32)
        for i, obs2 in zip(range(n), generator()):
            self.assertTrue(np.all(obs[i] == obs2))
        obs = np.zeros((n, NUM_ANGLES, NUM_DET_PIXELS), dtype=np.float32)
        obs_ = get_observations(range(n), out=obs)
        self.assertIs(obs_, obs)
        for i, obs2 in zip(range(n), generator()):
            self.assertTrue(np.all(obs[i] == obs2))
        # key spanning two files
        key = range(126, 130)
        obs = get_observations(key)
        for i, k in enumerate(key):
            obs2 = get_observation(k)
            self.assertTrue(np.all(obs[i] == obs2))
        with self.assertRaises(TypeError):
            get_observations([0, 1, 2])
        with self.assertRaises(ValueError):
            get_observations(range(0, 10, -1))
        with self.assertRaises(IndexError):
            get_observations(range(3679))
        with self.assertRaises(IndexError):
            get_observations(range(-1, 0))

class TestTransformToPreLog(unittest.TestCase):
    def test(self):
        obs = get_observation(0)
        obs_pre_log = transform_to_pre_log(obs, inplace=False)
        self.assertTrue(np.allclose(obs_pre_log, np.exp(-obs * MU_MAX)))
        obs2 = obs.copy()
        obs2_ = transform_to_pre_log(obs2)
        self.assertIs(obs2_, obs2)
        self.assertTrue(np.allclose(obs2, np.exp(-obs * MU_MAX)))
        obs2 = obs.asarray().copy()
        obs2_ = transform_to_pre_log(obs2)
        self.assertIs(obs2_, obs2)
        self.assertTrue(np.allclose(obs2, np.exp(-obs * MU_MAX)))

class TestReplaceMinPhotonCount(unittest.TestCase):
    def test(self):
        obs_0 = get_observation(0)  # sample 0 contains some zero photon counts
        obs_0_replaced = replace_min_photon_count(obs_0, 0.1, inplace=False)
        self.assertAlmostEqual(np.max(obs_0_replaced), np.max(obs_0))
        obs_0_replaced = replace_min_photon_count(obs_0, 0.5, inplace=False)
        self.assertLess(np.max(obs_0_replaced), np.max(obs_0))
        obs_0_2 = obs_0.copy()
        obs_0_2_ = replace_min_photon_count(obs_0_2, 0.1)
        self.assertIs(obs_0_2_, obs_0_2)
        self.assertAlmostEqual(np.max(obs_0_2), np.max(obs_0))
        obs_0_2 = obs_0.asarray().copy()
        replace_min_photon_count(obs_0_2, 0.5)
        self.assertLess(np.max(obs_0_2), np.max(obs_0))
        obs_1 = get_observation(1)  # sample 1 contains no zero photon counts
        obs_1_replaced = replace_min_photon_count(obs_1, 1., inplace=False)
        self.assertAlmostEqual(np.max(obs_1_replaced), np.max(obs_1))
        obs_0_pre_log = transform_to_pre_log(obs_0, inplace=False)
        obs_0_pre_log_replaced = replace_min_photon_count(
            obs_0_pre_log, 0.1, obs_is_post_log=False, inplace=False)
        self.assertAlmostEqual(np.min(obs_0_pre_log_replaced),
                               np.min(obs_0_pre_log))
        obs_0_pre_log_replaced = replace_min_photon_count(
            obs_0_pre_log, 0.5, obs_is_post_log=False, inplace=False)
        self.assertGreater(np.min(obs_0_pre_log_replaced),
                           np.min(obs_0_pre_log))

if __name__ == '__main__':
    unittest.main()
