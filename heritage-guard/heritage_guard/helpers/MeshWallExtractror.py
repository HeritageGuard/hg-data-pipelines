import pyvista as pv
import numpy as np
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import time


@dataclass
class MeshWallExtractorConfig:
    mesh_path = './lidar/Mesh/Decimated.obj'
    vertical_threshold = 0.1
    '''Deviation from vertical wall angle in radians'''
    angle_weight = 10000
    '''Adjust this to give more importance to angles of the walls when clustering'''
    centroid_weight = 0.001
    '''Adjust this to give less importance to location of the walls when clustering'''
    esp = 0.03
    '''esp parameter for DBSCAN'''
    min_samples = 20
    '''min_samples parameter for DBSCAN'''


class MeshWallExtractor:
    def __init__(self, config: MeshWallExtractorConfig):
        self.config = config

    @staticmethod
    def calculate_azimuthal_angle(normal):
        angle = np.arctan2(normal[1], normal[0])  # Assuming XZ plane
        return np.degrees(angle) % 360

    def get_wall_planes(self) :
        t1 = time.time()
        mesh = pv.read(self.config.mesh_path).decimate(0.75)
        if mesh.cell_normals is None:
            mesh.compute_normals(cell_normals=True, inplace=True)
        normals = mesh.cell_normals
        print(f'Normals: {time.time() - t1}')

        t2 = time.time()
        vertical_indices = np.where(np.abs(normals[:, 2]) < self.config.vertical_threshold)[0]
        vertical_normals = normals[vertical_indices]
        print(f'Vertical norms: {time.time() - t2}')

        t3 = time.time()
        full_centroids = mesh.cell_centers().points[vertical_indices]
        centroids_2d = np.delete(full_centroids, 2, 1)
        print(f'Centroids: {time.time() - t3}')

        t4 = time.time()
        angles = np.array([MeshWallExtractor.calculate_azimuthal_angle(n) for n in vertical_normals]).reshape(-1, 1)
        print(f'Angles: {time.time() - t4}')

        weighted_angles = angles * self.config.angle_weight
        weighted_centroids = centroids_2d * self.config.centroid_weight

        features = np.hstack((weighted_angles, weighted_centroids))

        # Normalize and scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        t5 = time.time()
        optics_clustering = DBSCAN(eps=self.config.esp, min_samples=20).fit(scaled_features)
        labels = optics_clustering.labels_
        print(f'DBSCAN: {time.time() - t5}')

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        results = []

        t6 = time.time()
        for label in set(labels):
            if label == -1:
                continue
            cluster_normals = vertical_normals[labels == label]
            centroid = full_centroids[labels == label]
            average_centroid = np.mean(centroid, axis=0)
            average_normal = np.mean(cluster_normals, axis=0)

            min_bound = np.min(centroid, axis=0)
            max_bound = np.max(centroid, axis=0)
            bounds = [min_bound[0], max_bound[0],
                      min_bound[1], max_bound[1],
                      min_bound[2], max_bound[2]]

            results.append((average_centroid, average_normal, bounds))
        print(f'Calculate planes: {time.time() - t6}')
        return results


