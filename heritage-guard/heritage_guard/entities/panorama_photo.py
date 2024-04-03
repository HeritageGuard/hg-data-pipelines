import uuid
from typing import List, Tuple
from PIL import Image
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pyvista as pv

from .detected_object import DetectedObject
from .orientation import Orientation
from .bbox import BBox
from ..CONSTANTS import X_ADJUST, Y_ADJUST

Point3D = Tuple[float, float, float]
Point2D = Tuple[int, int]
NpArray = np.ndarray


class PanoramaPhoto:
    """
      Class containing data about panorama photo and bounding boxes
    """
    detected_objects: List[DetectedObject] = []

    def __init__(self, file_name: str, time: float, center: Point3D, orientation: Orientation):


        self.file_name = file_name
        self.center = (center[0] - X_ADJUST, center[1] - Y_ADJUST, center[2])
        self.orientation = orientation
        self.time = time
        self.detected_objects = []
        self.heading_offset = 0
        try:
            image = Image.open(self.file_name)
        except FileNotFoundError:
            print("Could not read the image file.")
            return
        self.image_width = image.width
        self.image_height = image.height

    def add_detected_object(self, detected_object) -> None:
        self.detected_objects.append(detected_object)

    def bbox_center_to_3d(self, pixel: Point2D, sphere_radius=1):
        return PanoramaPhoto.bbox_center_to_3d_pure(
            pixel,
            sphere_radius,
            self.orientation.heading,
            self.orientation.pitch,
            self.orientation.roll,
            self.image_width,
            self.image_height,
            self.center
        )

    @staticmethod
    def bbox_center_to_3d_pure(pixel, sphere_radius, heading, pitch, roll, image_width, image_height, center):
        x, y = pixel

        # Convert degrees to radians
        heading = np.radians(heading + 90)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

        # Convert 2D coordinates to spherical coordinates
        theta = 2 * np.pi * (image_width - x) / image_width  # azimuthal angle
        phi = np.pi * y / image_height  # polar angle

        # Step 2: Convert spherical coordinates to Cartesian coordinates (assuming sphere radius = 1)
        X = np.sin(phi) * np.cos(theta)
        Y = np.sin(phi) * np.sin(theta)
        Z = np.cos(phi)

        # Step 4: Create rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])  # Roll
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])  # Pitch
        Rz = np.array(
            [[np.cos(heading), np.sin(heading), 0], [-np.sin(heading), np.cos(heading), 0], [0, 0, 1]])  # Heading (Yaw)

        # Combined rotation matrix: R = Rz * Ry * Rx (adjust as per the correct rotation order)
        R = np.dot(Rz, np.dot(Ry, Rx))

        # Apply rotation
        XYZ = np.dot(R, np.array([X, Y, Z])) + center

        def extend_line(point_on_sphere, t):
            direction_vector = point_on_sphere - center
            extended_point = center + t * direction_vector
            return extended_point

        return extend_line(XYZ, sphere_radius)

    @staticmethod
    def shift_seam(x, shift_amount, image_width):
        return (x + shift_amount) % image_width

    @staticmethod
    def shift_bbox_and_polygon(bbox, polygon, shift_amount, image_width):
        shifted_bbox = BBox(
            (PanoramaPhoto.shift_seam(bbox.x_min, shift_amount, image_width),
                bbox.y_min,
                PanoramaPhoto.shift_seam(bbox.x_max, shift_amount, image_width),
                bbox.y_max),
            image_width
        )

        # Extract x and y coordinates
        x_coords, y_coords = polygon.exterior.coords.xy
        # Shift x-coordinates
        shifted_x_coords = [(x + shift_amount) % image_width for x in x_coords]

        # Create a new shifted polygon
        shifted_polygon_points = list(zip(shifted_x_coords, y_coords))
        shifted_polygon = Polygon(shifted_polygon_points)

        return shifted_bbox, shifted_polygon

    def visualize(self, score_cutoff: float = 0, shift_amount: int = 0) -> None:
        try:
            image = Image.open(self.file_name)
        except FileNotFoundError:
            print("Could not read the image file.")
            return

        # Shift the image
        np_image = np.array(image)
        shifted_image = np.roll(np_image, shift_amount, axis=1)

        _, ax = plt.subplots(1)
        ax.imshow(shifted_image)
        ax.axis('off')
        for detected_object in self.detected_objects:
            if detected_object.score < score_cutoff:
                continue

            # Shift bbox and polygon
            shifted_bbox, shifted_polygon = PanoramaPhoto.shift_bbox_and_polygon(
                detected_object.bbox,
                detected_object.polygon,
                shift_amount,
                image.size[0]
            )

            color = np.random.rand(3)
            thickness = 2
            rect = patches.Rectangle(shifted_bbox.origin, shifted_bbox.width, shifted_bbox.height, linewidth=thickness,
                                     edgecolor=color, facecolor='none')
            poly = patches.Polygon(np.array(shifted_polygon.exterior.coords.xy).T, edgecolor=color, facecolor='none')
            ax.add_patch(poly)
            # ax.add_patch(rect)
            center_x, center_y = shifted_bbox.center

            ax.scatter(center_x, center_y, c=color, marker='o')
            # ax.text(shifted_bbox.origin[0], shifted_bbox.origin[1], f'{round(detected_object.score, 2)} - {detected_object.object_class.name}')
        plt.show()

    def visualize_sphere(self, plotter):
        sphere = pv.Sphere(
            radius=1,
            theta_resolution=120,
            phi_resolution=120,
            # center=(row[-2]-Y_ADJUST, row[-3]-X_ADJUST, row[-1]),
        )
        sphere.active_t_coords = np.zeros((sphere.points.shape[0], 2))

        for i in range(sphere.points.shape[0]):
            sphere.active_t_coords[i] = [
                0.5 + np.arctan2(-sphere.points[i, 0], sphere.points[i, 1]) / (2 * np.pi),
                0.5 + np.arcsin(sphere.points[i, 2]) / np.pi,
            ]

        heading_rad = self.orientation.rads.heading + 0.1
        heading_shift = np.array([-np.sin(heading_rad), 0])
        sphere.active_t_coords -= 0.5
        sphere.active_t_coords += heading_shift
        sphere.active_t_coords += 0.5

        sphere.translate(self.center, inplace=True)
        sphere.points[:] = (sphere.points - sphere.center) * 2 + sphere.center
        texture = pv.read_texture(self.file_name)

        return plotter.add_mesh(sphere, texture=texture, opacity=0.8, pickable=True)

    def visualize_3d(self, point_cloud: NpArray = None, score_cutoff: float = 0, plotter=None) -> None:
        p = plotter
        if p is None:
            p = pv.Plotter()

        if point_cloud:
            cloud = pv.PolyData(point_cloud)
            p.add_mesh(cloud, point_size=1, render_points_as_spheres=False)

        sphere = pv.Sphere(
            radius=1,
            theta_resolution=120,
            phi_resolution=120,
            # center=(row[-2]-Y_ADJUST, row[-3]-X_ADJUST, row[-1]),
        )
        sphere.active_t_coords = np.zeros((sphere.points.shape[0], 2))

        for i in range(sphere.points.shape[0]):
            sphere.active_t_coords[i] = [
                0.5 + np.arctan2(-sphere.points[i, 0], sphere.points[i, 1]) / (2 * np.pi),
                0.5 + np.arcsin(sphere.points[i, 2]) / np.pi,
            ]

        heading_rad = self.orientation.rads.heading + 0.1
        heading_shift = np.array([-np.sin(heading_rad), 0])
        sphere.active_t_coords -= 0.5
        sphere.active_t_coords += heading_shift
        sphere.active_t_coords += 0.5

        sphere.translate(self.center, inplace=True)
        sphere.points[:] = (sphere.points - sphere.center) * 2 + sphere.center
        texture = pv.read_texture(self.file_name)

        p.add_mesh(sphere, texture=texture, opacity=0.8)
        heading = self.orientation.heading
        pitch = self.orientation.pitch
        roll = self.orientation.roll
        image_width = self.image_width
        image_height = self.image_height
        center = self.center

        def draw_poly_lines(detected_object):
            if detected_object.score <= score_cutoff:
                return
            bbox = detected_object.bbox
            lines = []
            for poly in list(zip(*detected_object.polygon.exterior.coords.xy))[0::10]:
                lines.append(pv.Line(center, PanoramaPhoto.bbox_center_to_3d_pure(
                    poly,
                    30,
                    heading,
                    pitch,
                    roll,
                    image_width,
                    image_height,
                    center
                )))
            return lines

        poly_lines = [draw_poly_lines(i) for i in self.detected_objects]
        for poly_line in sum([x for x in poly_lines if x is not None], []):
            if poly_line is not None:
                p.add_mesh(poly_line, color='#04384d', opacity=0.5)

            # p.add_mesh(pv.Line(self.center, self.bbox_center_to_3d(bbox.center, sphere_radius=100)), color='red')
        if plotter is None:
            p.show()

    @staticmethod
    def extended_line(origin, point, distance):
        # Calculate the direction vector
        direction = (point[0] - origin[0], point[1] - origin[1])

        # Normalize the direction vector
        magnitude = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        direction = (direction[0] / magnitude, direction[1] / magnitude)

        # Extend the line
        extended_point = (origin[0] + direction[0] * distance, origin[1] + direction[1] * distance)
        return LineString([origin, extended_point])

    @staticmethod
    def extended_point_3d(origin, point, distance):
        # Calculate the direction vector (including z-coordinate)
        direction = (point[0] - origin[0], point[1] - origin[1], point[2] - origin[2])

        # Normalize the direction vector (including z-coordinate)
        magnitude = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
        direction = (direction[0] / magnitude, direction[1] / magnitude, direction[2] / magnitude)

        # Extend the line (including z-coordinate)
        extended_point = (origin[0] + direction[0] * distance,
                          origin[1] + direction[1] * distance,
                          origin[2] + direction[2] * distance)

        # Return a 3D line (assuming LineString can handle 3D points; if not, use an appropriate 3D line representation)
        return origin, extended_point

    def get_lines(self, score_cutoff=0):
        lines = []
        for detected_object in self.detected_objects:
            if detected_object.score < score_cutoff:
                continue
            bbox = detected_object.bbox
            bbox_center_3d = self.bbox_center_to_3d(bbox.center, sphere_radius=100)
            lines.append((PanoramaPhoto.extended_line(self.center[:2], bbox_center_3d[:2], 100), self.center, bbox_center_3d))

        return lines

    def get_polygon_lines(self, score_cutoff=0, spacing=1):
        print(self.file_name)
        lines = []
        for detected_object in self.detected_objects:
            if detected_object.score < score_cutoff:
                continue
            object_id = str(uuid.uuid4())
            for poly in list(zip(*detected_object.polygon.exterior.coords.xy))[0::spacing]:
                poly_3d = self.bbox_center_to_3d(poly, sphere_radius=100)
                lines.append((PanoramaPhoto.extended_line(self.center[:2], poly_3d[:2], 100), self.center, poly_3d, object_id))
        return lines

    def __str__(self):
        return f"""File name: {self.file_name},
    Gps time: {self.time},
    Centroid: x={self.center[0]}, y={self.center[1]}, z={self.center[2]},
    Orientation: roll={self.orientation.roll}, pitch={self.orientation.pitch}, heading={self.orientation.heading}"""
