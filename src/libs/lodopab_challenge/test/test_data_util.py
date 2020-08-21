# -*- coding: utf-8 -*-
import unittest
import os
import h5py
import numpy as np

from lodopab_challenge.data_util import read_h5_file

class TestReadH5File(unittest.TestCase):
    def setUp(self):
        self.H5FILENAME = 'test.hdf5'
        self.N = 4
        self.IM_SHAPE = (32, 32)
        self.reconstructions = [
            np.random.random(self.IM_SHAPE).astype(np.float32)
            for _ in range(self.N)]
        with h5py.File(self.H5FILENAME, 'w') as file:
            dataset = file.create_dataset(
                'data', shape=(self.N,) + self.IM_SHAPE,
                dtype=np.float32, fillvalue=np.nan, chunks=True)
            for i in range(self.N):
                dataset[i] = self.reconstructions[i]

    def test(self):
        out = read_h5_file(self.H5FILENAME)
        self.assertEqual(out.shape, (self.N,) + self.IM_SHAPE)
        self.assertTrue(np.all(out == np.asarray(self.reconstructions)))
        out2 = read_h5_file(self.H5FILENAME, n=self.N//2)
        self.assertEqual(out2.shape, (self.N//2,) + self.IM_SHAPE)
        self.assertTrue(
            np.all(out2 == np.asarray(self.reconstructions[:self.N//2])))
        out3 = np.zeros((self.N,) + self.IM_SHAPE, dtype=np.float32)
        out3_ = read_h5_file(self.H5FILENAME, out=out3)
        self.assertIs(out3, out3_)
        self.assertTrue(np.all(out3 == np.asarray(self.reconstructions)))
        out4 = np.zeros((self.N//2,) + self.IM_SHAPE, dtype=np.float32)
        out4_ = read_h5_file(self.H5FILENAME, out=out4, n=self.N//2)
        self.assertIs(out4, out4_)
        self.assertTrue(
            np.all(out4 == np.asarray(self.reconstructions[:self.N//2])))

    def tearDown(self):
        os.remove(self.H5FILENAME)

    def testExceptions(self):
        BAD_H5FILENAME = 'test_bad.hdf5'
        with h5py.File(BAD_H5FILENAME, 'w') as file:
            file.create_dataset(
                'data', shape=self.IM_SHAPE,  # missing first dim in shape
                dtype=np.float32, fillvalue=np.nan, chunks=True)
        out = np.zeros((self.N,) + self.IM_SHAPE, dtype=np.float32)
        with self.assertRaises(ValueError):
            read_h5_file(BAD_H5FILENAME)
        with self.assertRaises(ValueError):
            read_h5_file(BAD_H5FILENAME, n=self.N//2)
        with self.assertRaises(ValueError):
            read_h5_file(BAD_H5FILENAME, out=out)
        with self.assertRaises(ValueError):
            read_h5_file(BAD_H5FILENAME, out=out, n=self.N//2)
        os.remove(BAD_H5FILENAME)
        with self.assertRaises(ValueError):
            read_h5_file(self.H5FILENAME, n=self.N+1)
        out2 = np.zeros((self.N,) + self.IM_SHAPE, dtype=np.float32)
        with self.assertRaises(ValueError):
            read_h5_file(self.H5FILENAME, out=out2, n=self.N+1)
        out3 = np.zeros((self.N+1,) + self.IM_SHAPE, dtype=np.float32)
        with self.assertRaises(ValueError):
            read_h5_file(self.H5FILENAME, out=out3, n=self.N+1)
        out4 = np.zeros((self.N-1,) + self.IM_SHAPE, dtype=np.float32)
        with self.assertRaises(ValueError):
            read_h5_file(self.H5FILENAME, out=out4, n=self.N)


if __name__ == '__main__':
    unittest.main()
