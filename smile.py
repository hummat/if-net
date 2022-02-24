import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree

import models.local_model as model
from models.generation import Generator
import data_processing.implicit_waterproofing as iw
from data_processing.evaluation import eval_mesh, eval_pointcloud, get_threshold_percentage


if __name__ == "__main__":
    input_dim = 128  # 256 for SVR and 128 for PCD reconst.
    output_dim = 256
    batch_points = 250000  # 100000 max for SVR and 250000 max for PCD reconst.
    device = torch.device("cuda")
    net = model.ShapeNetPoints()

    mesh = trimesh.load("/home/matthias/Data/Ubuntu/data/aae_workspace/models/case.ply", process=False)
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)

    angle = 90 / 180 * np.pi
    R = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
    mesh.apply_transform(R)

    # mesh = mesh.slice_plane((0, 0, 0), (1, 0, 0))
    points = mesh.sample(3000).astype(np.float32)
    # trimesh.PointCloud(points).show()
    # np.random.shuffle(mesh.vertices)
    # pcd = mesh.vertices[:3000]

    grid_points = iw.create_grid_points_from_bounds(-0.5, 0.5, input_dim)
    kdtree = KDTree(grid_points)
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    _, idx = kdtree.query(points)
    occupancies[idx] = 1
    inputs = np.reshape(occupancies, (input_dim,) * 3).astype(np.float32)

    data = {"inputs": torch.unsqueeze(torch.from_numpy(inputs), dim=0)}

    gen = Generator(model=net,
                    threshold=0.5,
                    exp_name="ShapeNet3000Points",
                    checkpoint=23,
                    device=device,
                    resolution=output_dim,
                    batch_points=batch_points)

    logits = gen.generate_mesh(data)
    mesh_pred = gen.mesh_from_logits(logits)
    out_dict = eval_mesh(mesh_pred, mesh, -0.5, 0.5)

    thresholds = np.linspace(1. / 1000, 1, 1000)
    precision = get_threshold_percentage(out_dict["accuracy"], thresholds)
    recall = get_threshold_percentage(out_dict["completeness"], thresholds)
    F = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]
    print(F[9], F[14], F[19])
    # mesh.export("smile.off")
    # mesh.show()
