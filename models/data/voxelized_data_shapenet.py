from __future__ import division

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class VoxelizedDataset(Dataset):

    def __init__(self, mode, res=32, voxelized_pointcloud=False, pointcloud_samples=3000, data_path='shapenet/data/',
                 split_file='shapenet/split.npz',
                 batch_size=64, num_sample_points=1024, num_workers=12, sample_distribution=[1], sample_sigmas=[0.015],
                 **kwargs):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert not np.any(self.sample_distribution < 0)
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)

        self.data = self.split[mode]
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxelized_pointcloud = voxelized_pointcloud
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path + self.data[idx]

        if not self.voxelized_pointcloud:
            occupancies = np.load(path + '/voxelization_{}.npy'.format(self.res))
            occupancies = np.unpackbits(occupancies)
            input = np.reshape(occupancies, (self.res,) * 3)
        else:
            # voxel_path = path + '/voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)
            # occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
            # input = np.reshape(occupancies, (self.res,) * 3)

            # My data:
            # surface_data = np.load(os.path.join(path, "surface_grid.npy"))[:50000]  # DISN
            surface_data = np.load(os.path.join(path, "pointcloud.npz"))["points"]  # OccNet
            indices = np.random.randint(surface_data.shape[0], size=self.pointcloud_samples)
            points = surface_data[indices, :3]

            # Scale to from (-0.55, 0.55) to (0, 1) range
            points /= 1.1
            points += 0.5
            # points[points > 1] = 1.0
            # points[points < 0] = 0.0

            # project points into voxel space: [0, k)
            # convert voxel space to voxel indices (truncate decimals: 0.1 -> 0)
            voxel_indices = (points * self.res).astype(int)

            # Populate occupancy grid
            occupancies = np.zeros((self.res,) * 3, dtype=np.int8)
            occupancies[voxel_indices] = 1
            input = occupancies

        # points = []
        # coords = []
        # occupancies = []

        # for i, num in enumerate(self.num_samples):
        #     boundary_samples_path = path + '/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
        #     boundary_samples_npz = np.load(boundary_samples_path)
        #     boundary_sample_points = boundary_samples_npz['points']
        #     boundary_sample_coords = boundary_samples_npz['grid_coords']
        #     boundary_sample_occupancies = boundary_samples_npz['occupancies']
        #     subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
        #     points.extend(boundary_sample_points[subsample_indices])
        #     coords.extend(boundary_sample_coords[subsample_indices])
        #     occupancies.extend(boundary_sample_occupancies[subsample_indices])

        # My data:
        # DISN
        # points_data = np.load(os.path.join(path, "uniform_random.npy"))  # Closest to IF-Net: surface_random.npy
        # indices = np.random.randint(points_data.shape[0], size=self.num_sample_points)
        # points = points_data[indices, :3]
        # occupancies = (points_data[indices, 3] <= 0)

        # OccNet
        points_data = np.load(os.path.join(path, "points.npz"))
        points = points_data["points"]
        indices = np.random.randint(points.shape[0], size=self.num_sample_points)
        points = points[indices]
        occupancies = np.unpackbits(points_data["occupancies"])[indices]

        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        # IF-Net trained in (-0.5, 0.5) range (i.e. no padding)
        points += 0.55
        points /= 1.1
        points -= 0.5

        grid_coords = points.copy()
        # Axes swap needed for grid sampling.
        grid_coords[:, 0], grid_coords[:, 2] = points[:, 2], points[:, 0]
        coords = 2 * grid_coords  # Scaling needed for grid sampling

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        return {'grid_coords': np.array(coords, dtype=np.float32),
                'occupancies': np.array(occupancies, dtype=np.float32),
                'points': np.array(points, dtype=np.float32),
                'inputs': np.array(input, dtype=np.float32), 'path': path}

    def get_loader(self, shuffle=True):

        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn)

    @staticmethod
    def worker_init_fn(worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
